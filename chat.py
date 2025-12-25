"""
🏰 Interactive Chat Interface for Academic Burnout Advisor
Speak thy mind, noble scholar, and receive wise counsel!
"""

import requests
import os
import sys

# API Configuration
API_URL = "http://127.0.0.1:8000"

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the welcome banner"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║  🏰 ACADEMIC BURNOUT PREVENTION & ADVISORY SYSTEM 🏰              ║
║  ═══════════════════════════════════════════════════════════════   ║
║                                                                    ║
║  Hark, noble scholar! Share thy thoughts and feelings,             ║
║  and receive wise counsel from the Oracle of Wellness! ⚔️          ║
║                                                                    ║
║  Commands:                                                         ║
║    • Type thy journal entry and press Enter                        ║
║    • Type 'quit' or 'exit' to leave                                ║
║    • Type 'clear' to clear the screen                              ║
║    • Type 'help' for guidance                                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

def print_help():
    """Print help information"""
    print("""
📜 GUIDANCE FOR THE WEARY SCHOLAR:
══════════════════════════════════════════════════════════════════════

Simply type how you're feeling about your academic life, for example:

  ✦ "I'm feeling overwhelmed with all my assignments and can't sleep."
  ✦ "Had a great day! Finished my project and hung out with friends."
  ✦ "I don't know if I can keep going. Everything feels hopeless."

The Oracle shall divine thy mental state and bestow upon thee:
  🏷️  A classification (Healthy, Stressed, or Burnout)
  📊  Confidence scores
  💡  Personalized recommendations
  🆘  Emergency resources (if needed)

══════════════════════════════════════════════════════════════════════
    """)

def get_risk_color(label):
    """Return color indicator for risk level"""
    if label == "Healthy":
        return "🟢"
    elif label == "Stressed":
        return "🟡"
    else:
        return "🔴"

def analyze_entry(text):
    """Send journal entry to API and get analysis"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to server. Make sure the API is running!"}
    except Exception as e:
        return {"error": str(e)}

def display_result(result):
    """Display the analysis result beautifully"""
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    prediction = result.get("prediction", {})
    advice = result.get("advice", {})
    
    label = prediction.get("label", "Unknown")
    confidence = prediction.get("confidence", 0)
    probabilities = prediction.get("probabilities", {})
    risk_level = prediction.get("risk_level", "")
    
    color = get_risk_color(label)
    
    print("\n" + "═" * 70)
    print(f"🔮 THE ORACLE SPEAKS:")
    print("═" * 70)
    
    # Prediction
    print(f"\n{color} Thy Mental State: **{label.upper()}**")
    print(f"📊 Confidence: {confidence:.1%}")
    print(f"⚠️  Risk Level: {risk_level}")
    
    # Probabilities
    print(f"\n📈 Probability Distribution:")
    for state, prob in probabilities.items():
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   {state:10} [{bar}] {prob:.1%}")
    
    # Summary
    print(f"\n💬 {advice.get('summary', '')}")
    
    # Severity Score
    severity = advice.get('severity_score', 0)
    print(f"\n📉 Severity Score: {severity}/10")
    
    # Top Recommendations
    recommendations = advice.get("recommendations", [])
    if recommendations:
        print(f"\n💡 TOP RECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n   {rec.get('category', '')} {rec.get('title', '')}")
            print(f"   {rec.get('description', '')}")
            actions = rec.get('action_items', [])[:2]
            for action in actions:
                print(f"     • {action}")
    
    # Quick Tip
    print(f"\n{advice.get('quick_tip', '')}")
    
    # Emergency Resources (if burnout)
    emergency = advice.get("emergency_resources")
    if emergency:
        print("\n" + "🚨" * 25)
        print("🆘 EMERGENCY RESOURCES:")
        print("-" * 50)
        for line in emergency.get("crisis_lines", []):
            name = line.get("name", "")
            number = line.get("number", line.get("text", ""))
            print(f"   📞 {name}: {number}")
        print(f"\n   💜 {emergency.get('message', '')}")
        print("🚨" * 25)
    
    # Follow-up
    print(f"\n📅 {advice.get('follow_up', '')}")
    print("═" * 70)

def check_server():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main chat loop"""
    clear_screen()
    print_banner()
    
    # Check server
    print("🔍 Checking connection to the Oracle...")
    if not check_server():
        print("""
❌ The Oracle slumbers! The API server is not running.

To awaken the Oracle, open a new terminal and run:
    cd Burnout_Advisor_Project
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Then run this chat script again!
        """)
        input("Press Enter to exit...")
        return
    
    print("✅ The Oracle is awake and ready to receive thy words!\n")
    
    while True:
        print("\n" + "─" * 70)
        try:
            user_input = input("📝 Share thy thoughts, noble scholar:\n> ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Fare thee well, noble scholar! Take care of thyself! ⚔️")
            break
        except EOFError:
            break
        
        if not user_input:
            print("⚠️  Please share thy thoughts! The Oracle cannot divine from silence.")
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Fare thee well, noble scholar! Take care of thyself! ⚔️")
            break
        elif user_input.lower() == 'clear':
            clear_screen()
            print_banner()
            continue
        elif user_input.lower() == 'help':
            print_help()
            continue
        elif len(user_input) < 10:
            print("⚠️  Thy message is too brief! Please share more of thy thoughts (at least 10 characters).")
            continue
        
        # Analyze the entry
        print("\n🔮 The Oracle is divining thy mental state...")
        result = analyze_entry(user_input)
        display_result(result)

if __name__ == "__main__":
    main()
