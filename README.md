# ShotSweep

Find the screenshots hiding in your WhatsApp image folders, fully offline, so you can clean them up.

ShotSweep runs [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) zero-shot classification on your CPU. It scores each image against four groups and saves the results in a local SQLite database:

| Label          | What it catches                                         |
| -------------- | ------------------------------------------------------- |
| `screenshot`   | phone/desktop screenshots, chats, websites, app screens |
| `text_graphic` | memes, greeting forwards, posters, flyers               |
| `document`     | photos of paper, receipts, handwritten notes            |
| `photo`        | people, places, food, objects, events                   |

A Streamlit dashboard lets you browse the results, mark images as *delete* or *keep*, and export the marked paths.

**ShotSweep never moves or deletes files.** You decide what to do with the exported list.

## Install

Requires Python 3.10+.

```sh
pip install -r requirements.txt
```

The first run downloads the CLIP model (~600 MB) into the Hugging Face cache. After that, everything works offline. Set `HF_HUB_OFFLINE=1` to make sure nothing is fetched.

## Usage

By default ShotSweep looks for a `Whatsapp` folder next to the scripts and scans its `WhatsApp Images` and `GBWhatsApp Images` subfolders. To use a different location:

```sh
export WHATSAPP_DIR=/path/to/WhatsApp        # PowerShell: $env:WHATSAPP_DIR = "D:\path\to\WhatsApp"
```

1. **Embed** each image once. This is the slow step (roughly 10–25 images/s on a laptop CPU). It saves after every batch, so you can stop it and run it again to resume.

   ```sh
   python screenshot_sorter.py embed
   python screenshot_sorter.py embed --dirs "/some/other/folder"   # scan specific folders
   ```

2. **Classify**. This takes a few seconds and can be re-run any time, even while `embed` is still running:

   ```sh
   python screenshot_sorter.py classify
   ```

3. **Review** in the dashboard:

   ```sh
   python dashboard.py        # or: streamlit run dashboard.py
   ```

   - Filter by label, score range, folder, and mark status.
   - Mark single images, or everything in the current filter, as delete/keep.
   - Run classify from the sidebar.
   - Export the paths marked *delete* as a text file.

## Tips

- Check the scores before trusting them. Look through the borderline range (about 0.4–0.7) in the dashboard before you mark anything in bulk.
- To change what counts as a screenshot, edit the prompt lists in `LABELS` in `screenshot_sorter.py` and run `classify` again. You don't need to run `embed` again.
- All data stays on your machine in `screenshot_sorter.db`. Never commit this file: it contains file paths from your device.

## How it works

- Each image is padded to a square rather than center-cropped, so tall screenshots keep their status bar and app layout, which are the strongest screenshot cues.
- The model turns each image into a 512-dimensional embedding (a list of numbers describing it) and stores it as float16.
- Each label's prompts are converted the same way and averaged. The softmax over the four labels becomes each image's scores.
