"""Generate a small multi-page sample PDF for testing and demoing the
ingestion pipeline, so the repo doesn't depend on the user supplying their
own PDF just to try it out."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PAGES = [
    (
        "Company Travel Policy",
        [
            "All employees traveling for business must book flights at least 14 days in advance",
            "whenever possible. Economy class is standard for flights under 6 hours. Business",
            "class requires VP approval for flights over 6 hours. Employees must submit expense",
            "reports within 14 days of trip completion using the standard expense template.",
        ],
    ),
    (
        "Reimbursement Rules",
        [
            "Meal reimbursement is capped at $75 per day for domestic travel and $100 per day",
            "for international travel. Alcohol is not reimbursable under any circumstances.",
            "Hotel bookings should not exceed $250 per night without prior manager approval.",
            "Rideshare and taxi receipts must be itemized and attached to the expense report.",
        ],
    ),
    (
        "Remote Work Guidelines",
        [
            "Employees may work remotely up to 3 days per week with manager approval. Remote",
            "work requests must be submitted through the HR portal at least one week in advance.",
            "All remote employees must be reachable during core hours of 10am to 3pm local time.",
            "Equipment stipends of up to $500 are available annually for home office setup.",
        ],
    ),
]


def make_pdf(path: str) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    for title, lines in PAGES:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 720, title)
        c.setFont("Helvetica", 11)
        y = 690
        for line in lines:
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()


if __name__ == "__main__":
    make_pdf("sample_docs/employee_handbook.pdf")
    print("Wrote sample_docs/employee_handbook.pdf")
