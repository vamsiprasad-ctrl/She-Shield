# 🛡️ She‑Shield: AI-Powered Safety Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-orange.svg)](https://flask.palletsprojects.com/)
[![Project Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Made by Vamsi](https://img.shields.io/badge/Author-Vamsi%20Prasad-red.svg)](https://github.com/vamsiprasad-ctrl)

---

## 🌟 Overview

**She‑Shield** is an AI-driven safety assistant aimed at empowering users—especially women—with real-time emergency support and location-based safety features. It features...

- 🗺️ Live location tracking & safety scoring  
- 📞 One-click SOS alert to emergency contacts  
- 🤖 AI-powered advice for safety scenarios  
- 🔄 Clean, modular Flask/Python backend  
- 🎨 Lightweight and responsive frontend  

---

## 📚 Table of Contents

- [Features](#-features)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Screenshots](#-screenshots)  
- [Configuration](#-configuration)  
- [Security Notes](#-security-notes)  
- [License](#-license)  
- [Author](#-author)  

---

## ✨ Features

- 🗺️ **Geolocation-based safety scoring** using Google Safety API  
- 📞 **Emergency alert** via SMS or email (Twilio / SendGrid integration)  
- ⏰ **Background phone tracking** with periodic location updates  
- 📱 **Responsive web interface** for mobile access  
- 🔄 **Modular code structure** for easy extension  

---

## 📥 Installation

### 🔧 Prerequisites

- Python 3.10+  
- Google API key for Maps/Safety  
- Twilio account (SMS)
- SendGrid (email alerts)

---

### 🛠️ Quick Setup Instructions

```bash
git clone https://github.com/vamsiprasad-ctrl/She-Shield.git
cd She-Shield

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Configure environment variables
echo GOOGLE_API_KEY=your_google_api_key > .env
echo TWILIO_SID=your_twilio_sid >> .env
echo TWILIO_TOKEN=your_twilio_auth_token >> .env
echo SENDGRID_API_KEY=your_sendgrid_api_key >> .env

# Run the web server
python app.py
