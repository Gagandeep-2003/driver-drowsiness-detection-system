# Contributing to Driver Drowsiness Detection System 🚗💤

Thanks for your interest in contributing! This guide will help you get started.

🙏 **If you haven't already, please consider [starring the repository ⭐](https://github.com/Gagandeep-2003/Driver-Drowsiness-Detection-System)** to show your support and help others discover the project.

---

## 🛠️ Project Setup

1. **Fork the repository**  
   Click the `Fork` button on the top right of the repo.

2. **Clone your fork**  
   ```bash
   git clone https://github.com/Gagandeep-2003/driver-drowsiness-detection-system.git
   cd driver-drowsiness-detection-system

    Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Install dependencies

    pip install -r requirements.txt

Download the facial landmark model  

The dlib facial landmark model is required for blink detection. Download it from the link below, unzip it, and place the `shape_predictor_68_face_landmarks.dat` file inside the `/models` directory.
* **Download Link**: [shape_predictor_68_face_landmarks.dat](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

🔄 How to Contribute

    Check existing issues or ask to be assigned one on Discord.

    Create a new branch

git checkout -b feature-name

Make your changes

Commit your changes

git add .
git commit -m "Added feature XYZ"

Push and create a pull request (PR)

    git push origin feature-name

    Then go to your fork on GitHub and click "Compare & pull request".

✅ Guidelines

    Keep your code clean and readable

    Follow existing naming and code style

    Comment your code where needed

    One feature or bug fix per PR

🙌 Need Help?

Feel free to ask questions in the GSSoC Discord or open a discussion!

Let’s build something amazing together 🚀


---

Let me know if you want to add sections like testing instructions or folder structure too!
