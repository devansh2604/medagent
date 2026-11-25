"""Private S3 report archive for a single MedAgent operator."""
import hashlib
import hmac
import os
from uuid import UUID

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

archive = Blueprint("archive", __name__)
LIMIT = 10 * 1024 * 1024


def enabled():
    return bool(os.getenv("MEDAGENT_REPORT_BUCKET") and os.getenv("MEDAGENT_ARCHIVE_TOKEN"))


@archive.route("/api/reports/<patient_id>", methods=["GET", "PUT"])
@archive.route("/api/reports/<patient_id>/<uuid:report_id>", methods=["GET"])
def reports(patient_id, report_id=None):
    import agent

    if current_app.config["DEMO_MODE"] or not enabled():
        return jsonify(error="Report archive disabled."), 404
    expected = "Bearer " + os.environ["MEDAGENT_ARCHIVE_TOKEN"]
    if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), expected.encode()):
        return jsonify(error="Archive access denied."), 403
    if patient_id not in agent.PATIENT_STORE:
        return jsonify(error="Patient record no longer exists."), 404

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    from botocore.config import Config

    bucket = os.environ["MEDAGENT_REPORT_BUCKET"]
    # Existing records predate archive IDs. New registrations get a fresh namespace
    # even if the operator reuses a deleted patient ID.
    identity = agent.PATIENT_STORE[patient_id].get("archive_id") or "legacy:" + patient_id
    prefix = "reports/" + hashlib.sha256(identity.encode()).hexdigest() + "/"
    try:
        s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
        if request.method == "PUT":
            try:
                upload_id = UUID(request.headers.get("X-Report-ID", ""))
            except ValueError:
                return jsonify(error="Missing report ID."), 400
            if request.content_length is None or request.content_length > LIMIT:
                return jsonify(error="Reports must be 10 MiB or smaller."), 413
            data = request.stream.read(LIMIT + 1)
            formats = [(b"%PDF-", "application/pdf"), (b"\x89PNG\r\n\x1a\n", "image/png"),
                       (b"\xff\xd8\xff", "image/jpeg")]
            mime = next((mime for signature, mime in formats if data.startswith(signature)), None)
            if not mime:
                return jsonify(error="Choose a PDF, PNG or JPEG report."), 415
            name = secure_filename(request.headers.get("X-Report-Name", ""))[:160] or "report"
            key = f"{prefix}{upload_id}/{name}"
            s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=mime,
                          ContentDisposition=f'attachment; filename="{name}"',
                          ServerSideEncryption="AES256")
            return jsonify(saved=True), 201

        search = prefix if report_id is None else f"{prefix}{report_id}/"
        items = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=search):
            for obj in page.get("Contents", []):
                if report_id is not None:
                    url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": obj["Key"]},
                                                    ExpiresIn=60)
                    return jsonify(url=url), 200, {"Cache-Control": "no-store"}
                identifier, name = obj["Key"][len(prefix):].split("/", 1)
                items.append(dict(id=identifier, name=name, saved=obj["LastModified"].isoformat()))
        if report_id is not None:
            return jsonify(error="Report not found."), 404
        return jsonify(sorted(items, key=lambda item: item["saved"], reverse=True)), 200, {"Cache-Control": "no-store"}
    except (BotoCoreError, ClientError):
        return jsonify(error="S3 is unavailable. Check the bucket, region and archive role."), 503
