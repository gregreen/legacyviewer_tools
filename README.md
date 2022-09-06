legacyviewer\_tools
===================

Code to generate mosaics and videos from the Legacy Survey Sky Browser.

Example usage
-------------

First, generate a JSON containing one WCS dictionary for each frame:

    python gen_framespec.py -c0 301.65071841 -6.12318614 28 -c1 301.65071841 -6.12318614 0.5 -n 600 --coordsys galactic --resolution 480p --layers decaps2[2] decaps2[1] decaps2[0] > musca_480p.json

Optionally, set a directory in which to store the cutouts downloaded from the Legacy Survey Sky Browser. The cutouts will be cached here, so that they do not have to be downloaded twice. This also means that the directory may fill up over time:

    export CUTOUT_DIR="/path/to/cutout/directory/"

If `CUTOUT_DIR` is not set, then it defaults to `cutouts/`.

Then, feed the frame specification JSON to `gen_video.py` to generate the frames:

    python3 gen_video.py \
        musca_480p.json \
        --img-outpattern frames/musca_gamma0.25_480p_frame{:05d}.png \
        --reproject-method interp \
        --momentum 0.95 \
        --gamma 0.25

Finally, turn these frames into a video, using `ffmpeg`:

    ffmpeg -y -r 30 -i frames/musca_gamma0.25_480p_frame%05d.png \
        -c:v libx264 -crf 22 -pix_fmt yuv420p -r 30 \
        videos/musca_gamma0.25_480p.mp4

