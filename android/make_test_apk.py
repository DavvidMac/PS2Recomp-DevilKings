"""Builds a debuggable test APK around the runner library without Gradle.

The app is a NativeActivity with no Java code, so the SDK's build tools are enough:
aapt2 compiles the manifest, the library goes in lib/arm64-v8a, zipalign aligns it and
apksigner signs with a debug key (created with the JDK's keytool if missing).

usage: python make_test_apk.py <libps2EntryRunner.so> <output.apk> [--label NAME]
Environment: ANDROID_SDK (default %LOCALAPPDATA%/Android/Sdk), JAVA_HOME for keytool.
"""
import argparse
import os
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
    <application android:label="{label}" android:hasCode="false" android:isGame="true"
        android:debuggable="true" android:extractNativeLibs="true">
        <activity android:name="android.app.NativeActivity"
            android:configChanges="orientation|screenSize|screenLayout|keyboard|keyboardHidden|navigation"
            android:screenOrientation="landscape" android:exported="true">
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('library')
    parser.add_argument('output')
    parser.add_argument('--label', default='Devil Kings (teste)')
    args = parser.parse_args()

    sdk = os.environ.get('ANDROID_SDK') or os.path.join(os.environ['LOCALAPPDATA'], 'Android', 'Sdk')
    tools = newest(os.path.join(sdk, 'build-tools'))
    platform = newest(os.path.join(sdk, 'platforms'))
    exe = '.exe' if os.name == 'nt' else ''
    bat = '.bat' if os.name == 'nt' else ''

    work = tempfile.mkdtemp(prefix='ps2x-apk-')
    manifest = os.path.join(work, 'AndroidManifest.xml')
    with open(manifest, 'w', encoding='utf-8') as out:
        out.write(MANIFEST.format(label=args.label))
    base = os.path.join(work, 'base.apk')
    run(os.path.join(tools, 'aapt2' + exe), 'link', '-o', base, '--manifest', manifest,
        '-I', os.path.join(platform, 'android.jar'))

    # The library uncompressed, so zipalign -p can page-align it.
    with zipfile.ZipFile(base, 'a') as apk:
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
