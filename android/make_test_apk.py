"""Builds a debuggable test APK around the runner library without Gradle.

The app is a NativeActivity plus one small Java class (ImmersiveActivity.java, which hides the
system bars), so the SDK's build tools and a JDK are enough: javac and d8 make classes.dex, aapt2
compiles the manifest, the library goes in lib/arm64-v8a, zipalign aligns it and apksigner signs
with a debug key (created with the JDK's keytool if missing).

usage: python make_test_apk.py <libps2EntryRunner.so> <output.apk> [--label NAME] [--icon FILE]
--icon takes a PNG, or a Windows .ico with PNG images (the largest is used; the runner's
PS2X_RUNNER_ICON works as is), and becomes the launcher icon; the default is PS2X_ANDROID_ICON.
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
        <activity android:name="com.ps2x.runner.ImmersiveActivity"
            android:configChanges="orientation|screenSize|screenLayout|keyboard|keyboardHidden|navigation"
            android:screenOrientation="fullUser" android:exported="true">
            <meta-data android:name="android.app.lib_name" android:value="ps2EntryRunner" />
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
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
        out.write(MANIFEST.format(label=args.label, icon=' android:icon="@mipmap/ic_launcher"' if args.icon else ''))
    base = os.path.join(work, 'base.apk')
    run(os.path.join(tools, 'aapt2' + exe), 'link', '-o', base, '--manifest', manifest,
        '-I', os.path.join(platform, 'android.jar'), *resources)

    # The activity class: javac against the platform's android.jar, then d8 to classes.dex.
    jdk_bin = os.path.join(os.environ['JAVA_HOME'], 'bin') if os.environ.get('JAVA_HOME') else ''
    classes = os.path.join(work, 'classes')
    os.makedirs(classes)
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ImmersiveActivity.java')
    run(os.path.join(jdk_bin, 'javac' + exe), '--release', '11', '-nowarn', '-classpath', os.path.join(platform, 'android.jar'),
        '-d', classes, source)
    class_files = [os.path.join(root, name) for root, _, names in os.walk(classes) for name in names if name.endswith('.class')]
    run(os.path.join(tools, 'd8' + bat), '--min-api', '28', '--lib', os.path.join(platform, 'android.jar'),
        '--output', work, *class_files)

    # The library uncompressed, so zipalign -p can page-align it.
    with zipfile.ZipFile(base, 'a') as apk:
        apk.write(os.path.join(work, 'classes.dex'), 'classes.dex')
        apk.write(args.library, 'lib/arm64-v8a/libps2EntryRunner.so', compress_type=zipfile.ZIP_STORED)
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
