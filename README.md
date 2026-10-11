# College FAQ Chatbot

A beginner-friendly, local chatbot that answers college questions by finding
the closest question in a predefined FAQ collection. It uses NLTK to
preprocess text and scikit-learn's TF-IDF and cosine similarity to choose an
answer. It does not use paid APIs, translation services, or large language
models.

## Features

- 24 college FAQs covering admissions, courses, tuition and fees, office and
  class times, examinations, student services, and contact information.
- FAQ content stored separately in `faqs.json`.
- Lowercase conversion, punctuation removal, common filler-word filtering,
  and NLTK tokenization.
- TF-IDF question matching with cosine similarity.
- Adjustable similarity threshold and a helpful fallback for unmatched
  questions.
- Streamlit chat history, empty-input validation, and a clear-chat button.
- FAQ-file and FAQ-data validation with readable error messages.
- Uses NLTK's bundled tokenizer, so no NLTK corpus download is needed.

## Project structure

```text
college-faq-chatbot/
├── app.py
├── faqs.json
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9 or newer
- Visual Studio Code

## Installation

1. Open this project folder in VS Code.
2. Open the integrated terminal using **Terminal > New Terminal**.
3. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

4. Activate it in the VS Code PowerShell terminal:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell prevents activation, use Command Prompt in VS Code and run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

5. Install the required libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. In VS Code, choose the virtual environment as the Python interpreter:
   press **Ctrl+Shift+P**, select **Python: Select Interpreter**, then choose
   the interpreter inside `.venv`.

## Run the chatbot

From the project folder, with the virtual environment activated, run:

```powershell
python -m streamlit run app.py
```

Streamlit opens the application in your browser. If it does not open
automatically, use the `Local URL` printed in the terminal (usually
`http://localhost:8501`).

## Try these sample questions

- How do I apply for admission?
- What documents do I need for an application?
- Which courses does the college offer?
- How much is tuition?
- Are scholarships available?
- What time does the library close?
- Where can I find my exam timetable?
- How do I contact the admissions office?

Try an unrelated question, such as "How do I repair a bicycle?", to see the
fallback response. Adjust **Similarity threshold** in the sidebar to make FAQ
matching stricter or more permissive. Select **Clear chat** to remove the
conversation history.

## Customize the FAQs

Edit `faqs.json` to use your college's real information. Keep it as a JSON
list, with each entry containing non-empty `question` and `answer` strings.
The included address, email addresses, and telephone numbers are sample
placeholders, not real college contact details.
