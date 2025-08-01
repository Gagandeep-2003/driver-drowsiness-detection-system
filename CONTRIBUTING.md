# Contributing to Driver Drowsiness Detection System 🚗💤

First off, thank you for considering contributing to this project! Your help is greatly appreciated.

🙏 **If you haven't already, please consider starring the [repository](https://github.com/Gagandeep-2003/Driver-Drowsiness-Detection-System) ⭐ to show your support and help others discover the project.**

---

## 🛠️ Project Setup
Follow these steps to get the project running on your local machine.

### 1. **Fork and Clone**
1.  **Fork the repository**: Click the `Fork` button on the top right of the repo.
2.  **Clone your fork**:

    ```bash
    git clone https://github.com/YOUR-USERNAME/driver-drowsiness-detection-system.git
    cd driver-drowsiness-detection-system
    ```

### 2. **Set Up Your Environment**
1.  **Create a virtual environment** (optional but highly recommended):
    
    ```bash
    # On Windows
    py -m venv venv
    venv\Scripts\activate
    
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
    
2.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dlib facial landmark model**  
    The blink detection feature requires a pre-trained model.
    * Download the file from [this link](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    * Unzip it and place the `shape_predictor_68_face_landmarks.dat` file inside the `/models` directory.

<br>

## 🔄 How to Contribute

1.  **Find an issue**: Check the existing [issues](https://github.com/Gagandeep-2003/Driver-Drowsiness-Detection-System/issues) to see if you can help, or raise a new issue if you have a new idea.
2.  **Create a new branch**:
    
    ```bash
    git checkout -b your-feature-name
    ```
    
3.  **Make your changes**: Write your code and make sure to follow the project's style.
4.  **Commit your changes**:
    
    ```bash
    git add .
    git commit -m "feat: Add feature XYZ"
    ```
    
5.  **Push and create a pull request (PR)**:
    
    ```bash
    git push origin your-feature-name
    ```
    
    Then, go to your fork on GitHub and click "Compare & pull request".

---

## ✅ Contribution Guidelines
* Keep your code clean, readable, and well-commented where necessary.
* Follow the existing naming conventions and code style.
* Ensure each pull request addresses only one feature or bug fix.
* Comment your code where needed.

## 🙌 Need Help?

Feel free to ask questions in the GSSoC Discord or open a discussion!

> Let’s build something amazing together 🚀
