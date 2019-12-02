import boto3
import zipfile
import os
from typing import *
import util


def zip_save(zipname: str, compfiles: Union[str, List[str]], timestamp: bool = False) -> str:
    assert compfiles
    if timestamp:
        zipname = util.get_time_hash() + "_" + zipname
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as new_zip:
        for pos, dirs, files in os.walk("."):
            if "archive" in pos:
                continue
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == ".py":
                    new_zip.write(os.path.join(pos, file), arcname=os.path.join("archive", pos, file))
        if isinstance(compfiles, str):
            compfiles = [compfiles]
        for file in compfiles:
            basename = os.path.basename(file)
            new_zip.write(file, arcname=basename)
    return zipname


def upload(zipname: str) -> None:
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("tokudono-rnn2wfa")
    bucket.upload_file(zipname, zipname)

def zip_and_upload(zipname: str, compfiles: Union[str, List[str]], timestamp: bool = False, skip_upload=False) -> str:
    """
    Save the given files (compfiles) into a zip file and uploads to our S3 bucket.
    The python files in the root is also saved in "archive" directory for the later debugging.
    :param zipname:
    :param compfiles:
    :return:
    """
    zipname = zip_save(zipname, compfiles, timestamp)
    if not skip_upload:
        upload(zipname)
    else:
        print("skipped uploading")
    return zipname


def download(zipname: str) -> None:
    if not os.path.exists(zipname):
        s3 = boto3.resource("s3")
        bucket = s3.Bucket("tokudono-rnn2wfa")
        bucket.download_file(zipname, zipname)


if __name__ == "__main__":
    zip_and_upload("test.zip", "README.md")
    download("pyonly.zip")
