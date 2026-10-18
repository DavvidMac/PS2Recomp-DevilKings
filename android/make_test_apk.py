"""Builds a debuggable test APK around the runner library without Gradle.

The app is a NativeActivity plus two small Java classes, so the SDK's build tools and a JDK are
enough: javac and d8 make classes.dex, aapt2 compiles the manifest, the library goes in
lib/arm64-v8a, zipalign aligns it and apksigner signs with a debug key (created with the JDK's
keytool if missing). SetupActivity.java is the launcher entry: it starts the game
(ImmersiveActivity.java, which hides the system bars) when the game's files are in the app's
folder, and otherwise asks for the game's .iso and copies them there once.

usage: python make_test_apk.py <libps2EntryRunner.so> <output.apk> [--label NAME] [--icon FILE]
                               [--boot-elf NAME] [--data DIR]... [--game-iso FILE]
--icon takes a PNG, or a Windows .ico with PNG images (the largest is used; the runner's
PS2X_RUNNER_ICON works as is), and becomes the launcher icon; the default is PS2X_ANDROID_ICON.
--boot-elf is the game's ELF in the .iso (default SLUS_212.97, Devil Kings): it tells whether the
game is installed and which .iso to accept. Each --data folder goes in the APK and is copied into
the game's folder on start where its files are missing (e.g. --data iso/cheats). --game-iso puts
the game's files (read from its .iso) in the APK, uncompressed under assets/game, and the app copies
them out on its first start instead of asking for the .iso: a personal copy, about 3 GB for Devil
Kings (the APK has to stay under 4 GB, as apksigner does not take ZIP64).
Environment: ANDROID_SDK (default %LOCALAPPDATA%/Android/Sdk), JAVA_HOME for javac and keytool.
"""
import argparse
import os
import struct
import shutil
import subprocess
import sys
import tempfile
import zipfile

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.ps2x.runner" android:versionCode="1" android:versionName="0.1.0">
    <uses-sdk android:minSdkVersion="28" android:targetSdkVersion="34" />
    <uses-feature android:glEsVersion="0x00030000" android:required="true" />
    <application android:label="{label}"{icon} android:hasCode="true" android:isGame="true"
        android:debuggable="true" android:extractNativeLibs="true">
        <activity android:name="com.ps2x.runner.SetupActivity"
            android:theme="@android:style/Theme.DeviceDefault.NoActionBar"
            android:configChanges="orientation|screenSize|screenLayout|keyboard|keyboardHidden|navigation"
            android:exported="true">
            <meta-data android:name="ps2x.boot_elf" android:value="{boot_elf}" />
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        <activity android:name="com.ps2x.runner.ImmersiveActivity"
            android:configChanges="orientation|screenSize|screenLayout|keyboard|keyboardHidden|navigation"
            android:screenOrientation="fullUser" android:exported="true">
            <meta-data android:name="android.app.lib_name" android:value="ps2EntryRunner" />
        </activity>
    </application>
</manifest>
"""


def newest(directory):
    versions = sorted(os.listdir(directory), key=lambda v: [int(p) if p.isdigit() else p for p in v.replace('-', '.').split('.')])
    return os.path.join(directory, versions[-1])


def run(*command):
    subprocess.run(command, check=True)


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
ISO_SECTOR = 2048


def iso_files(path):
    """(path, byte offset, size) of every file in an ISO 9660 image, ';1' versions dropped."""
    out = []
    with open(path, 'rb') as image:
        def read(sector, size):
            image.seek(sector * ISO_SECTOR)
            return image.read(size)

        volume = read(16, ISO_SECTOR)
        if volume[0] != 1 or volume[1:6] != b'CD001':
            raise SystemExit(f'{path}: not an ISO 9660 image')

        def walk(sector, size, prefix):
            data = read(sector, size)
            position = 0
            while position < len(data):
                length = data[position]
                if length == 0:
                    position = (position // ISO_SECTOR + 1) * ISO_SECTOR
                    continue
                record = data[position:position + length]
                position += length
                extent, = struct.unpack_from('<I', record, 2)
                bytes_, = struct.unpack_from('<I', record, 10)
                name = record[33:33 + record[32]]
                if name in (b'\x00', b'\x01'):
                    continue
                name = name.decode('ascii').split(';')[0]
                if record[25] & 2:
                    walk(extent, bytes_, prefix + name + '/')
                else:
                    out.append((prefix + name, extent * ISO_SECTOR, bytes_))

        root_sector, = struct.unpack_from('<I', volume, 156 + 2)
        root_size, = struct.unpack_from('<I', volume, 156 + 10)
        walk(root_sector, root_size, '')
    return out


def add_game(apk_path, iso):
    """Appends the .iso's files to the APK, stored, under assets/game."""
    files = iso_files(iso)
    total = os.path.getsize(apk_path) + sum(size + 128 for _, _, size in files)
    if total >= 0xFFFFFFFF - (64 << 20):
        raise SystemExit(f'the APK would take {total / 1e9:.2f} GB: over the 4 GB a ZIP without ZIP64 holds')
    # zipfile switches to ZIP64 past 2 GB, which apksigner refuses; a plain ZIP holds 4 GB.
    zipfile.ZIP64_LIMIT = 0xFFFFFFFF - 1
    with open(iso, 'rb') as image, zipfile.ZipFile(apk_path, 'a') as apk:
        for name, offset, size in files:
            info = zipfile.ZipInfo('assets/game/' + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.file_size = size
            image.seek(offset)
            with apk.open(info, 'w') as out:
                left = size
                while left:
                    chunk = image.read(min(left, 16 << 20))
                    if not chunk:
                        raise SystemExit(f'{iso}: ends inside {name}')
                    out.write(chunk)
                    left -= len(chunk)
            print(f'  game: {name} ({size / 1e6:.1f} MB)')


def icon_png(path):
    """The PNG bytes of an icon file: the file itself, or the largest PNG image of an .ico."""
    data = open(path, 'rb').read()
    if data.startswith(PNG_SIGNATURE):
        return data
    reserved, kind, count = struct.unpack_from('<HHH', data, 0)
    if reserved != 0 or kind != 1:
        raise SystemExit(f'{path}: not a PNG or an .ico')
    best = None
    for index in range(count):
        width, height, _, _, _, _, size, offset = struct.unpack_from('<BBBBHHII', data, 6 + index * 16)
        image = data[offset:offset + size]
        side = width or 256
        if image.startswith(PNG_SIGNATURE) and (best is None or side > best[0]):
            best = (side, image)
    if best is None:
        raise SystemExit(f'{path}: the .ico has no PNG image')
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('library')
    parser.add_argument('output')
    parser.add_argument('--label', default='Devil Kings (teste)')
    parser.add_argument('--icon', default=os.environ.get('PS2X_ANDROID_ICON'))
    parser.add_argument('--boot-elf', default='SLUS_212.97')
    parser.add_argument('--data', action='append', default=[])
    parser.add_argument('--game-iso')
    args = parser.parse_args()

    sdk = os.environ.get('ANDROID_SDK') or os.path.join(os.environ['LOCALAPPDATA'], 'Android', 'Sdk')
    tools = newest(os.path.join(sdk, 'build-tools'))
    platform = newest(os.path.join(sdk, 'platforms'))
    exe = '.exe' if os.name == 'nt' else ''
    bat = '.bat' if os.name == 'nt' else ''

    work = tempfile.mkdtemp(prefix='ps2x-apk-')
    # The launcher icon: one image at the highest density; Android scales it for the others.
    resources = []
    if args.icon:
        folder = os.path.join(work, 'res', 'mipmap-xxxhdpi')
        os.makedirs(folder)
        with open(os.path.join(folder, 'ic_launcher.png'), 'wb') as out:
            out.write(icon_png(args.icon))
        compiled = os.path.join(work, 'resources.zip')
        run(os.path.join(tools, 'aapt2' + exe), 'compile', '--dir', os.path.join(work, 'res'), '-o', compiled)
        resources.append(compiled)
    manifest = os.path.join(work, 'AndroidManifest.xml')
    with open(manifest, 'w', encoding='utf-8') as out:
        out.write(MANIFEST.format(label=args.label, boot_elf=args.boot_elf,
                                  icon=' android:icon="@mipmap/ic_launcher"' if args.icon else ''))
    # Files copied into the game's folder on start (assets/data/<folder name>/...).
    assets = []
    for folder in args.data:
        shutil.copytree(folder, os.path.join(work, 'assets', 'data', os.path.basename(os.path.normpath(folder))))
    if args.data:
        assets = ['-A', os.path.join(work, 'assets')]
    base = os.path.join(work, 'base.apk')
    run(os.path.join(tools, 'aapt2' + exe), 'link', '-o', base, '--manifest', manifest,
        '-I', os.path.join(platform, 'android.jar'), *assets, *resources)

    # The activity class: javac against the platform's android.jar, then d8 to classes.dex.
    jdk_bin = os.path.join(os.environ['JAVA_HOME'], 'bin') if os.environ.get('JAVA_HOME') else ''
    classes = os.path.join(work, 'classes')
    os.makedirs(classes)
    here = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(here, name) for name in ('ImmersiveActivity.java', 'SetupActivity.java')]
    run(os.path.join(jdk_bin, 'javac' + exe), '--release', '11', '-nowarn', '-classpath', os.path.join(platform, 'android.jar'),
        '-d', classes, *sources)
    class_files = [os.path.join(root, name) for root, _, names in os.walk(classes) for name in names if name.endswith('.class')]
    run(os.path.join(tools, 'd8' + bat), '--min-api', '28', '--lib', os.path.join(platform, 'android.jar'),
        '--output', work, *class_files)

    # The library uncompressed, so zipalign -p can page-align it.
    with zipfile.ZipFile(base, 'a') as apk:
        apk.write(os.path.join(work, 'classes.dex'), 'classes.dex')
        apk.write(args.library, 'lib/arm64-v8a/libps2EntryRunner.so', compress_type=zipfile.ZIP_STORED)
    if args.game_iso:
        add_game(base, args.game_iso)
    aligned = os.path.join(work, 'aligned.apk')
    run(os.path.join(tools, 'zipalign' + exe), '-p', '-f', '4', base, aligned)

    keystore = os.path.join(os.path.expanduser('~'), '.android', 'debug.keystore')
    if not os.path.exists(keystore):
        os.makedirs(os.path.dirname(keystore), exist_ok=True)
        keytool = os.path.join(os.environ['JAVA_HOME'], 'bin', 'keytool' + exe) if os.environ.get('JAVA_HOME') else 'keytool'
        run(keytool, '-genkeypair', '-keystore', keystore, '-alias', 'androiddebugkey', '-storepass', 'android',
            '-keypass', 'android', '-keyalg', 'RSA', '-validity', '10000', '-dname', 'CN=Android Debug,O=Android,C=US')
    run(os.path.join(tools, 'apksigner' + bat), 'sign', '--ks', keystore, '--ks-pass', 'pass:android',
        '--key-pass', 'pass:android', '--out', args.output, aligned)
    shutil.rmtree(work, ignore_errors=True)
    print('apk:', args.output)


if __name__ == '__main__':
    sys.exit(main())
