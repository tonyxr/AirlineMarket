"""Download official TranStats form extracts with anonymous ASP.NET sessions."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import zipfile

from download import ROOT, RECORDS, atomic, describe, digest, inventory, now


class Form(HTMLParser):
    """Read actual hidden values, data checkboxes, select options and lookup links."""

    def __init__(self):
        super().__init__()
        self.hidden, self.fields, self.options, self.links = {}, [], {}, []
        self.select = None

    def handle_starttag(self, tag, attrs):
        """Capture only published form inputs; no guessed field identifiers."""
        a = dict(attrs)
        name = a.get("name", "")
        if tag == "input" and a.get("type") == "hidden" and name.startswith("__"):
            self.hidden[name] = a.get("value", "")
        if (
            tag == "input"
            and a.get("type") == "checkbox"
            and name
            and not name.startswith("chk")
        ):
            self.fields.append(name)
        if tag == "select":
            self.select = name
            self.options[name] = []
        if tag == "option" and self.select:
            self.options[self.select].append(a.get("value", ""))
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])

    def handle_endtag(self, tag):
        """Stop attributing options to a finished select."""
        if tag == "select":
            self.select = None


def curl(args, output, cookie, timeout=7200):
    """Use normal TLS and session cookies; fail on HTTP errors."""
    command = [
        "curl",
        "-fLSs",
        "--connect-timeout",
        "25",
        "--max-time",
        str(timeout),
        "-b",
        str(cookie),
        "-c",
        str(cookie),
        "-o",
        str(output),
        *args,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def fetch_form(job):
    """Select the requested actual year/period and preserve a complete ZIP."""
    dest = ROOT / job["path"]
    receipt = RECORDS / "files" / (job["id"] + ".json")
    if receipt.exists() and dest.exists():
        old = json.loads(receipt.read_text())
        if (
            old.get("status") == "downloaded"
            and old.get("bytes") == dest.stat().st_size
        ):
            return old
    directory = RECORDS / "forms" / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cookie = directory / "cookies.txt"
    form_path = directory / "form.html"
    partial = dest.with_suffix(dest.suffix + ".part")
    record = {
        **job,
        "source_url": job["url"],
        "started_at": now(),
        "status": "downloading",
        "rows": None,
        "date_range": None,
        "method": "anonymous official TranStats form, actual published fields selected",
    }
    atomic(receipt, record)
    try:
        if shutil.disk_usage(dest.parent).free < 8 * 1024**3:
            raise RuntimeError("storage reserve reached")
        curl([job["url"]], form_path, cookie, 120)
        parser = Form()
        parser.feed(form_path.read_text(errors="replace"))
        if not parser.fields:
            raise ValueError("no published data field checkboxes")
        data = dict(parser.hidden)
        for select, options in parser.options.items():
            if select == "cboYear":
                value = str(job["year"])
            elif select == "cboPeriod":
                value = str(job.get("period", "All"))
            elif select == "cboGeography":
                value = "All"
            else:
                continue
            if value not in options:
                raise ValueError(
                    f"{select}={value} is unavailable; published options={options}"
                )
            data[select] = value
        data.update({name: "on" for name in parser.fields})
        data.update(chkDownloadZip="on", btnDownload="Download")
        post = directory / "post.txt"
        post.write_text(urlencode(data))
        record.update(
            selected_fields=parser.fields,
            selection={k: v for k, v in data.items() if k.startswith("cbo")},
            lookup_urls=sorted(
                set(
                    urljoin(job["url"], link)
                    for link in parser.links
                    if "Download_Lookup" in link
                )
            ),
        )
        atomic(receipt, record)
        if not dest.exists():
            curl(["--data-binary", "@" + str(post), job["url"]], partial, cookie)
            if not zipfile.is_zipfile(partial):
                response = Form()
                response.feed(partial.read_text(errors="replace"))
                links = [
                    urljoin(job["url"], link)
                    for link in response.links
                    if ".zip" in link.lower()
                ]
                if len(links) != 1:
                    raise ValueError(
                        "form response is not a ZIP and does not expose one archive link"
                    )
                curl([links[0]], partial, cookie)
                record["generated_download_url"] = links[0]
            members = describe(partial, ".zip")
            partial.replace(dest)
        else:
            members = describe(dest)
        record.update(
            status="downloaded",
            completed_at=now(),
            bytes=dest.stat().st_size,
            sha256=digest(dest),
            members=members,
            integrity="Archive directory/member headers verified; streaming row/CRC validation pending",
        )
        print(
            f"downloaded {job['id']}: {dest.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )
    except Exception as error:
        record.update(status="failed", completed_at=now(), reason=str(error))
        print(f"failed {job['id']}: {error}", flush=True)
    atomic(receipt, record)
    inventory()
    return record


def main():
    """Serialize form jobs to avoid submitting duplicate server-side extracts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    name = Path(args.manifest).stem
    RECORDS.mkdir(parents=True, exist_ok=True)
    with (RECORDS / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = json.loads(Path(args.manifest).read_text())["jobs"]
        for index, job in enumerate(jobs):
            atomic(
                RECORDS / (name + ".status.json"),
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "active_job": job["id"],
                    "completed_jobs": index,
                    "configured_jobs": len(jobs),
                    "updated_at": now(),
                },
            )
            fetch_form(job)
        atomic(
            RECORDS / (name + ".status.json"),
            {
                "status": "finished",
                "pid": os.getpid(),
                "completed_jobs": len(jobs),
                "configured_jobs": len(jobs),
                "updated_at": now(),
            },
        )


if __name__ == "__main__":
    main()
