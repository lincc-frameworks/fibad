downloadCutout.py
==============================================================================

Download FITS cutouts from the website of HSC data release.

Requirements
------------------------------------------------------------------------------

python >= 3.7

Usage
------------------------------------------------------------------------------

### Download images of all bands at a location

```
python3 downloadCutout.py --ra=222.222 --dec=44.444 --sw=0.5arcmin --sh=0.5arcmin --name="cutout-{filter}"
```

Note that `{filter}` must appear in `--name`.
Otherwise, the five images of the five bands will be written
to a single file over and over.

### Use coordinate list

You can feed a coordinate list that is in nearly the same format as
https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/manual.html#list-to-upload

There are a few differences:

  - There must not appear comments
    except for the mandatory one at the first line.

  - You can use "all" as a value of "filter" field.

  - There may be columns with unrecognised names,
    which are silently ignored.

It is permissible for the coordinate list to contain only coordinates.
For example:

```
#? ra      dec
222.222  44.444
222.223  44.445
222.224  44.446
```

In this case, you have to specify other fields via the command line:

```
python3 downloadCutout.py \
    --sw=5arcsec --sh=5arcsec \
    --image=true --variance=true --mask=true \
    --name="cutout_{tract}_{ra}_{dec}_{filter}" \
    --list=coordlist.txt # <- the name of the above list
```

It is more efficient to use a list like the example above
than to use a for-loop to call the script iteratively.

### Stop asking a password

To stop the script asking your password, put the password
into an environment variable. (Default: `HSC_SSP_CAS_PASSWORD`)

```
read -s HSC_SSP_CAS_PASSWORD
export HSC_SSP_CAS_PASSWORD
```

Then, run the script with `--user` option:

```
python3 downloadCutout.py \
    --ra=222.222 --dec=44.444 --sw=0.5arcmin --sh=0.5arcmin \
    --name="cutout-{filter}" \
    --user=USERNAME
```

If you are using your own personal laptop or desktop,
you may pass your password through `--password` option.
But you must never do so
if there are other persons using the same computer.
Remember that other persons can see your command lines
with, for example, `top` command.
(If it is GNU's `top`, press `C` key to see others' command lines).

### Synchronize processes

If you run a program in parallel which calls `downloadCutout.py` sporadically
but frequently, the program needs synchronizing---the server refuses
`downloadCutout.py` if many instances of which are run at the same time.

If your program does not have a synchronization mechanism,
you can run `downloadCutout.py` with synchronization options:

```
python3 downloadCutout.py .... \
    --semaphore=/home/yourname/semaphore --max-connections=4
```

Because the processes synchronize with each other via the specified semaphore
(this is not a posix semaphore but a hand-made semaphore-like object),
the semaphore must be seen to all the processes.
If the processes are distributed over a network,
the semaphore must be placed in an NFS or any other shared filesystem.

Usage as a python module
------------------------------------------------------------------------------

Here is an example:

```
import downloadCutout

rect = downloadCutout.Rect.create(
    ra="11h11m11.111s",
    dec="-1d11m11.111s",
    sw="1arcmin",
    sh="1arcmin",
)

images = downloadCutout.download(rect)

# Multiple images (of various filters) are returned.
# We look into the first one of them.
metadata, data = images[0]
print(metadata)

# `data` is just the binary data of a FITS file.
# You can use, for example, `astropy` to decode it.
import io
import astropy.io.fits
hdus = astropy.io.fits.open(io.BytesIO(data))
print(hdus)
```
