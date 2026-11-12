# KotOR Mesh Upscaler

A Windows tool that adds real geometry to the low-poly models of **Star Wars: Knights of the Old Republic** and **KotOR II: The Sith Lords** (PC binary `.mdl` / `.mdx`). It smooths blocky silhouettes, can turn a normal map into actual surface detail, and writes a model the game loads directly. Animations, bones and everything else in the file are kept.

> Screenshot: original vs. upscaled (split view) — *add `docs/split-view.png` here*

## Features

- **Native MDL/MDX reader and writer.** No PyKotor needed. New data is appended and the mesh headers are repointed, so animations, controllers, bones, dangly/saber/walkmesh nodes and every other pointer stay valid.
- **Feature-aware smoothing.** Crease and hard-edge detection, smoothing-group normals, crack-free curved (PN-triangle) tessellation. UVs, normals and skin weights are interpolated.
- **Displacement from the normal map.** Integrates the existing `_n` normal map into a height field and adds it as real geometry (optional, per mesh). Can also carve cavities found in the diffuse texture.
- **Face budget mode.** Spread a total face count across the selected meshes by area.
- **UV regions.** Mark parts of a texture to smooth more, leave alone, or keep flat.
- **Checks.** UV health report with repair, head-proportion check against anatomy presets, and model validation (catches the size/offset mistakes that crash the game).
- **Safe install.** Validates the model, backs up anything it replaces (`.bak`), copies to your Override folder and verifies by MD5.
- **FBX round trip.** Export to binary FBX, edit in Blender, import back; skin weights are transferred automatically.
- **Built-in 3D viewer.** Orbit, pan, zoom, wireframe, original / upscaled / split view.

## Download

Get `KotorMeshUpscaler.exe` from the [Releases](../../releases) page. No install needed.

## Usage

1. Extract the model you want (`.mdl` **and** its `.mdx`) into one folder, for example with KotOR Tool or Holocron Toolset.
2. Open the `.mdl` in the upscaler (or drag it onto the `.exe`).
3. Select the meshes to upscale, adjust the sliders, click **Preview upscale** and compare in split view.
4. **Save upscaled model...** (a `.bak` is kept of anything overwritten), then **Validate model** and **Safe install to Override...**
5. Fully restart the game.

Viewer: drag to orbit, right-drag to pan, mouse wheel to zoom.

Notes:
- ASCII models are not supported; extract them as binary.
- Meshes with extra vertex channels (tangents, vertex colours, second UV set) are skipped for now.
- One mesh can hold at most 65,535 vertices (a limit of the format).

## Running from source

Requires Python 3.8+ (tested with 3.12) on Windows.

```
pip install -r requirements.txt
python kotor_mesh_upscaler.py [model.mdl]
```

## Building the .exe

```
pip install -r requirements.txt
pyinstaller --noconfirm KotorMeshUpscaler.spec
```

The executable is written to `dist\KotorMeshUpscaler.exe`.

## Support

If this tool saves you time, you can [buy me a coffee](https://www.buymeacoffee.com/YOUR-NAME). ☕

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

This project contains no game files. Star Wars: Knights of the Old Republic and its assets belong to their respective owners; only use it on models from your own copy of the game, and do not redistribute game files.
