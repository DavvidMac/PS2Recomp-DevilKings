package com.ps2x.runner;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.os.StatFs;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The launcher entry. The runner reads the game from the app's own folder
 * (Android/data/com.ps2x.runner/files), which only the app (or a PC over USB) can fill and which
 * Android empties when the app is uninstalled. When the game is there this starts
 * ImmersiveActivity at once. When it is not, the game's files are copied there once: from the
 * APK's assets/game when the APK carries them, otherwise from the game's .iso, which the player
 * picks through the system file picker (so no storage permission is needed).
 *
 * The manifest's "ps2x.boot_elf" meta-data names the game's ELF (SLUS_212.97 for Devil Kings);
 * an image whose SYSTEM.CNF boots another one is refused. Files under assets/data (the cheat
 * lists) are copied to the same folder when missing.
 */
public class SetupActivity extends Activity {
    private static final int PICK_ISO = 1;
    private static final int SECTOR = 2048;
    /** Present while a copy runs: a game folder that has it is incomplete. */
    private static final String IMPORTING = ".importando";
    private static final String BUNDLED = "game";

    private File gameDir;
    private String bootElf;
    private TextView message;
    private Button choose;
    private ProgressBar progress;
    private volatile boolean copying;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        gameDir = getExternalFilesDir(null);
        bootElf = "SLUS_212.97";
        try {
            Bundle meta = getPackageManager().getActivityInfo(getComponentName(), android.content.pm.PackageManager.GET_META_DATA).metaData;
            if (meta != null && meta.getString("ps2x.boot_elf") != null)
                bootElf = meta.getString("ps2x.boot_elf");
        } catch (Exception ignored) {
        }
        // An .iso handed over in the intent (e.g. adb shell am start -d file:///...iso) is imported
        // like one picked from the screen.
        final Uri handed = getIntent() != null ? getIntent().getData() : null;
        if (gamePresent() && handed == null) {
            startGame();
            return;
        }
        buildUi();
        if (handed != null)
            startCopy(handed);
        else if (hasBundledGame())
            startCopy(null);
        else
            showIntro();
    }

    private boolean gamePresent() {
        return gameDir != null && new File(gameDir, bootElf).isFile() && !new File(gameDir, IMPORTING).exists();
    }

    private boolean hasBundledGame() {
        try {
            String[] files = getAssets().list(BUNDLED);
            return files != null && files.length > 0;
        } catch (IOException e) {
            return false;
        }
    }

    private void startGame() {
        copyBundledData();
        startActivity(new Intent(this, ImmersiveActivity.class));
        finish();
    }

    // ------------------------------------------------------------------ screen

    private int dp(float value) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, getResources().getDisplayMetrics());
    }

    private void buildUi() {
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setGravity(Gravity.CENTER_HORIZONTAL);
        column.setPadding(dp(24), dp(32), dp(24), dp(24));
        column.setBackgroundColor(Color.rgb(18, 14, 14));

        TextView title = new TextView(this);
        title.setText(getApplicationInfo().loadLabel(getPackageManager()));
        title.setTextColor(Color.rgb(220, 40, 40));
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 30);
        title.setGravity(Gravity.CENTER);
        column.addView(title);

        message = new TextView(this);
        message.setTextColor(Color.rgb(230, 230, 230));
        message.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        message.setPadding(0, dp(20), 0, dp(20));
        message.setLineSpacing(0, 1.15f);
        column.addView(message);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(1000);
        progress.setVisibility(View.GONE);
        column.addView(progress, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(24)));

        choose = new Button(this);
        choose.setOnClickListener(v -> {
            if (hasBundledGame())
                startCopy(null);
            else
                pickIso();
        });
        LinearLayout.LayoutParams buttonLayout = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        buttonLayout.topMargin = dp(12);
        column.addView(choose, buttonLayout);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(18, 14, 14));
        scroll.addView(column);
        setContentView(scroll);
    }

    private void showIntro() {
        message.setText("Os arquivos do jogo ainda não estão neste aparelho.\n\n"
                        + "Copie o .iso do jogo para o celular (por exemplo, para a pasta Download), toque no botão "
                        + "abaixo e escolha o arquivo. O jogo é copiado para o app uma vez só (cerca de 3 GB livres "
                        + "são necessários); depois disso o .iso pode ser apagado.\n\n"
                        + "Desinstalar o app apaga essa cópia.");
        choose.setText("Escolher o arquivo .iso");
        choose.setVisibility(View.VISIBLE);
        progress.setVisibility(View.GONE);
    }

    private void pickIso() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_ISO);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == PICK_ISO && result == RESULT_OK && data != null && data.getData() != null)
            startCopy(data.getData());
    }

    /** Copies the game in the background: from the .iso at uri, or from the APK when uri is null. */
    private void startCopy(final Uri uri) {
        copying = true;
        choose.setVisibility(View.GONE);
        progress.setVisibility(View.VISIBLE);
        progress.setProgress(0);
        message.setText(uri == null ? "Preparando o jogo (só na primeira vez)..." : "Lendo o arquivo...");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        new Thread(() -> {
            String failure = null;
            try {
                if (uri == null)
                    importBundled();
                else
                    importIso(uri);
            } catch (Exception e) {
                failure = e.getMessage() != null ? e.getMessage() : e.toString();
            }
            final String error = failure;
            runOnUiThread(() -> {
                copying = false;
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                if (error == null) {
                    startGame();
                } else if (uri == null) {
                    progress.setVisibility(View.GONE);
                    message.setText("Não deu para preparar o jogo:\n" + error);
                    choose.setText("Tentar de novo");
                    choose.setVisibility(View.VISIBLE);
                } else {
                    showIntro();
                    message.setText("Não deu para copiar o jogo:\n" + error + "\n\nEscolha o .iso do jogo de novo.");
                }
            });
        }, "copiar-jogo").start();
    }

    @Override
    public void onBackPressed() {
        if (!copying)
            super.onBackPressed();
    }

    private void status(final String text, final long done, final long total) {
        runOnUiThread(() -> {
            message.setText(text);
            progress.setProgress(total > 0 ? (int) (done * 1000 / total) : 0);
        });
    }

    private void status(String path, long done, long total, boolean fromIso) {
        status(String.format(Locale.ROOT, "%s\n%s\n%.2f de %.2f GB",
                             fromIso ? "Copiando o jogo para o app..." : "Preparando o jogo (só na primeira vez)...",
                             path, done / 1e9, total / 1e9), done, total);
    }

    /** Checks the space and marks the folder incomplete until finishCopy(). */
    private File beginCopy(long total) throws IOException {
        if (gameDir == null)
            throw new IOException("o armazenamento do aparelho não está disponível");
        long free = new StatFs(gameDir.getPath()).getAvailableBytes();
        if (free < total + (64L << 20))
            throw new IOException(String.format(Locale.ROOT, "falta espaço: o jogo precisa de %.1f GB e há %.1f GB livres",
                                                total / 1e9, free / 1e9));
        File marker = new File(gameDir, IMPORTING);
        if (!marker.exists() && !marker.createNewFile())
            throw new IOException("não consegui escrever na pasta do app");
        return marker;
    }

    private static void finishCopy(File marker) throws IOException {
        if (!marker.delete())
            throw new IOException("não consegui terminar a cópia");
    }

    private File targetFor(String path) throws IOException {
        File target = new File(gameDir, path);
        File parent = target.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs())
            throw new IOException("não consegui criar " + parent.getName());
        return target;
    }

    // ------------------------------------------------------------------ from the APK

    private static void listAssets(AssetManager assets, String path, List<String> out) throws IOException {
        String[] children = assets.list(path);
        if (children == null || children.length == 0) {
            out.add(path);
            return;
        }
        for (String child : children)
            listAssets(assets, path + "/" + child, out);
    }

    private static long assetLength(AssetManager assets, String path) {
        try (AssetFileDescriptor descriptor = assets.openFd(path)) {
            return descriptor.getLength();
        } catch (IOException e) {
            return 0; // compressed: not counted
        }
    }

    private void importBundled() throws IOException {
        AssetManager assets = getAssets();
        List<String> files = new ArrayList<>();
        listAssets(assets, BUNDLED, files);
        long total = 0;
        for (String path : files)
            total += assetLength(assets, path);
        File marker = beginCopy(total);
        byte[] buffer = new byte[4 << 20];
        long done = 0;
        for (String path : files) {
            String relative = path.substring(BUNDLED.length() + 1);
            try (InputStream input = assets.open(path, AssetManager.ACCESS_STREAMING);
                 FileOutputStream output = new FileOutputStream(targetFor(relative))) {
                int count;
                while ((count = input.read(buffer)) > 0) {
                    output.write(buffer, 0, count);
                    done += count;
                    status(relative, done, total, false);
                }
            }
        }
        finishCopy(marker);
    }

    // ------------------------------------------------------------------ from an ISO 9660 image

    private static final class Entry {
        final String path;
        final long sector;
        final long size;

        Entry(String path, long sector, long size) {
            this.path = path;
            this.sector = sector;
            this.size = size;
        }
    }

    private static ByteBuffer read(FileChannel channel, long sector, int size) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        long position = sector * SECTOR;
        while (buffer.hasRemaining()) {
            int count = channel.read(buffer, position + buffer.position());
            if (count < 0)
                throw new IOException("o arquivo terminou antes do esperado (o .iso está incompleto?)");
        }
        buffer.flip();
        return buffer;
    }

    private static void walk(FileChannel channel, long sector, int size, String prefix, List<Entry> out, int depth) throws IOException {
        if (depth > 16)
            throw new IOException("pastas aninhadas demais no .iso");
        ByteBuffer data = read(channel, sector, size);
        int position = 0;
        while (position < size) {
            int length = data.get(position) & 0xFF;
            if (length == 0) {
                position = (position / SECTOR + 1) * SECTOR;
                continue;
            }
            long extent = data.getInt(position + 2) & 0xFFFFFFFFL;
            long bytes = data.getInt(position + 10) & 0xFFFFFFFFL;
            int flags = data.get(position + 25) & 0xFF;
            int nameLength = data.get(position + 32) & 0xFF;
            byte[] raw = new byte[nameLength];
            for (int i = 0; i < nameLength; ++i)
                raw[i] = data.get(position + 33 + i);
            position += length;
            if (nameLength == 1 && (raw[0] == 0 || raw[0] == 1))
                continue; // the folder itself and its parent
            String name = new String(raw, StandardCharsets.US_ASCII);
            int version = name.indexOf(';');
            if (version >= 0)
                name = name.substring(0, version);
            if (name.isEmpty() || name.contains("/") || name.contains("\\") || name.equals("..") || name.equals("."))
                throw new IOException("nome de arquivo inválido no .iso");
            if ((flags & 2) != 0)
                walk(channel, extent, (int) bytes, prefix + name + "/", out, depth + 1);
            else
                out.add(new Entry(prefix + name, extent, bytes));
        }
    }

    private void importIso(Uri uri) throws IOException {
        try (ParcelFileDescriptor descriptor = getContentResolver().openFileDescriptor(uri, "r");
             FileInputStream input = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = input.getChannel()) {
            ByteBuffer volume = read(channel, 16, SECTOR);
            byte[] magic = new byte[5];
            for (int i = 0; i < 5; ++i)
                magic[i] = volume.get(1 + i);
            if (volume.get(0) != 1 || !"CD001".equals(new String(magic, StandardCharsets.US_ASCII)))
                throw new IOException("esse arquivo não é um .iso de DVD (ISO 9660)");
            long rootSector = volume.getInt(156 + 2) & 0xFFFFFFFFL;
            int rootSize = volume.getInt(156 + 10);
            List<Entry> entries = new ArrayList<>();
            walk(channel, rootSector, rootSize, "", entries, 0);

            // The right game: SYSTEM.CNF has to boot the ELF this app runs.
            Entry config = null;
            long total = 0;
            for (Entry entry : entries) {
                total += entry.size;
                if (entry.path.equalsIgnoreCase("SYSTEM.CNF"))
                    config = entry;
            }
            if (config == null || config.size > 4096)
                throw new IOException("esse .iso não é de um jogo de PS2 (falta o SYSTEM.CNF)");
            ByteBuffer configData = read(channel, config.sector, (int) config.size);
            byte[] configBytes = new byte[configData.remaining()];
            configData.get(configBytes);
            String configText = new String(configBytes, StandardCharsets.US_ASCII);
            if (!configText.toUpperCase(Locale.ROOT).contains(bootElf.toUpperCase(Locale.ROOT)))
                throw new IOException("esse .iso é de outro jogo; este app roda o " + bootElf + ".\n" + configText.trim());

            File marker = beginCopy(total);
            byte[] buffer = new byte[4 << 20];
            ByteBuffer chunk = ByteBuffer.wrap(buffer);
            long done = 0;
            for (Entry entry : entries) {
                try (FileOutputStream output = new FileOutputStream(targetFor(entry.path))) {
                    long position = entry.sector * SECTOR;
                    long left = entry.size;
                    while (left > 0) {
                        chunk.clear();
                        chunk.limit((int) Math.min(buffer.length, left));
                        int count = channel.read(chunk, position);
                        if (count < 0)
                            throw new IOException("o arquivo terminou antes do esperado (o .iso está incompleto?)");
                        output.write(buffer, 0, count);
                        position += count;
                        left -= count;
                        done += count;
                        status(entry.path, done, total, true);
                    }
                }
            }
            finishCopy(marker);
        }
    }

    // ------------------------------------------------------------------ bundled small files

    /** Copies assets/data/** into the game folder where a file of the same name is missing. */
    private void copyBundledData() {
        try {
            copyAssets(getAssets(), "data", gameDir);
        } catch (IOException ignored) {
        }
    }

    private static void copyAssets(AssetManager assets, String path, File target) throws IOException {
        String[] children = assets.list(path);
        if (children == null || children.length == 0) {
            if (target.exists())
                return;
            File parent = target.getParentFile();
            if (parent != null)
                parent.mkdirs();
            try (InputStream input = assets.open(path); OutputStream output = new FileOutputStream(target)) {
                byte[] buffer = new byte[64 << 10];
                int count;
                while ((count = input.read(buffer)) > 0)
                    output.write(buffer, 0, count);
            }
            return;
        }
        for (String child : children)
            copyAssets(assets, path + "/" + child, new File(target, child));
    }
}
