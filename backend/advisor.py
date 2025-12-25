"""
Rule-Based Advisory System for Academic Burnout Prevention
Provides personalized recommendations based on burnout prediction results
"""

from typing import Dict, List
from enum import Enum
from dataclasses import dataclass
import random


class BurnoutLevel(Enum):
    """Enumeration of burnout levels"""
    HEALTHY = 0
    STRESSED = 1
    BURNOUT = 2


@dataclass
class Recommendation:
    """Data class for a recommendation"""
    category: str
    title: str
    description: str
    priority: int  # 1 = highest priority
    action_items: List[str]


# ============================================
# RECOMMENDATION DATABASE
# ============================================

RECOMMENDATIONS = {
    BurnoutLevel.HEALTHY: [
        Recommendation(
            category="🎯 Maintenance",
            title="Keep Up the Great Work!",
            description="Your mental health indicators look positive. Continue with your current healthy habits.",
            priority=1,
            action_items=[
                "Maintain your current sleep schedule",
                "Continue regular exercise routine",
                "Keep connecting with friends and family",
                "Consider helping a classmate who might be struggling"
            ]
        ),
        Recommendation(
            category="📈 Growth",
            title="Build Resilience for the Future",
            description="Now is a great time to develop habits that will protect you during stressful periods.",
            priority=2,
            action_items=[
                "Start a gratitude journal",
                "Learn a new stress-management technique (meditation, breathing exercises)",
                "Create an emergency self-care plan for stressful times",
                "Build a support network you can rely on"
            ]
        ),
        Recommendation(
            category="🌟 Enrichment",
            title="Pursue Personal Growth",
            description="With your mental health in a good place, explore opportunities for personal development.",
            priority=3,
            action_items=[
                "Take on a new hobby or project",
                "Mentor other students",
                "Join clubs or organizations aligned with your interests",
                "Set meaningful personal goals beyond academics"
            ]
        )
    ],
    
    BurnoutLevel.STRESSED: [
        Recommendation(
            category="⚡ Immediate Action",
            title="Address Current Stressors",
            description="You're showing signs of stress. Taking action now can prevent burnout.",
            priority=1,
            action_items=[
                "Identify your top 3 stressors and write them down",
                "Break large tasks into smaller, manageable steps",
                "Say no to non-essential commitments this week",
                "Schedule at least 30 minutes of 'you time' daily"
            ]
        ),
        Recommendation(
            category="😴 Rest & Recovery",
            title="Prioritize Sleep and Rest",
            description="Quality rest is essential for managing stress effectively.",
            priority=1,
            action_items=[
                "Aim for 7-8 hours of sleep tonight",
                "Create a relaxing bedtime routine (no screens 1 hour before bed)",
                "Take short breaks every 45-50 minutes while studying",
                "Practice deep breathing for 5 minutes when feeling overwhelmed"
            ]
        ),
        Recommendation(
            category="📅 Time Management",
            title="Reorganize Your Schedule",
            description="Better time management can significantly reduce academic stress.",
            priority=2,
            action_items=[
                "Use a planner or digital calendar for all deadlines",
                "Apply the Pomodoro Technique (25 min work, 5 min break)",
                "Prioritize tasks using the Eisenhower Matrix (urgent vs important)",
                "Batch similar tasks together to improve efficiency"
            ]
        ),
        Recommendation(
            category="🤝 Support",
            title="Seek Support",
            description="You don't have to handle everything alone. Reach out for help.",
            priority=2,
            action_items=[
                "Talk to a friend or family member about how you're feeling",
                "Visit your professor's office hours for academic help",
                "Join a study group to share the workload",
                "Consider speaking with a campus counselor"
            ]
        ),
        Recommendation(
            category="🏃 Physical Wellness",
            title="Move Your Body",
            description="Physical activity is a powerful stress reducer.",
            priority=3,
            action_items=[
                "Take a 15-minute walk between classes",
                "Do simple stretches at your desk",
                "Try a campus fitness class or gym session",
                "Dance to your favorite music for 10 minutes"
            ]
        )
    ],
    
    BurnoutLevel.BURNOUT: [
        Recommendation(
            category="🚨 Critical Priority",
            title="Seek Professional Help Immediately",
            description="Your responses indicate severe burnout. Professional support is strongly recommended.",
            priority=1,
            action_items=[
                "Contact your campus counseling center TODAY",
                "If in crisis, call a mental health helpline",
                "Talk to a trusted adult (parent, professor, advisor)",
                "Consider seeing a healthcare provider about physical symptoms"
            ]
        ),
        Recommendation(
            category="🛑 Stop & Reset",
            title="Give Yourself Permission to Pause",
            description="Recovery requires stepping back. Your health matters more than any deadline.",
            priority=1,
            action_items=[
                "Take a mental health day if possible",
                "Email professors about deadline extensions - most will understand",
                "Clear your schedule of all non-essential activities",
                "Allow yourself to rest without guilt"
            ]
        ),
        Recommendation(
            category="🌿 Basic Self-Care",
            title="Focus on Fundamentals",
            description="Start with the basics: eating, sleeping, and basic hygiene.",
            priority=1,
            action_items=[
                "Eat at least 3 meals today, even if small",
                "Go to bed at a reasonable hour tonight",
                "Take a shower and get dressed in fresh clothes",
                "Spend 10 minutes outside in natural light"
            ]
        ),
        Recommendation(
            category="💔 Emotional Support",
            title="You Are Not Alone",
            description="Many students experience burnout. There is no shame in struggling.",
            priority=2,
            action_items=[
                "Remind yourself that this is temporary",
                "Connect with at least one person today",
                "Write down 3 small things you accomplished recently",
                "Practice self-compassion - talk to yourself like you would a friend"
            ]
        ),
        Recommendation(
            category="📋 Academic Recovery",
            title="Create a Recovery Plan",
            description="Once you've stabilized, plan your academic recovery with support.",
            priority=3,
            action_items=[
                "Meet with your academic advisor about options",
                "Consider dropping a class if overloaded (check withdrawal deadlines)",
                "Explore incomplete grades or extensions",
                "Plan a reduced workload for next semester"
            ]
        )
    ]
}

# ============================================
# ADVISOR CLASS
# ============================================

class BurnoutAdvisor:
    """
    Rule-based advisory system for academic burnout prevention
    """
    
    def __init__(self):
        """Initialize the advisor"""
        self.recommendations = RECOMMENDATIONS
    
    def get_recommendations(self, prediction: Dict) -> Dict:
        """
        Get personalized recommendations based on prediction
        
        Args:
            prediction: Dictionary from BurnoutPredictor containing label_id, confidence, etc.
            
        Returns:
            Dictionary containing recommendations and advice
        """
        label_id = prediction.get("label_id", 0)
        confidence = prediction.get("confidence", 0.5)
        
        try:
            burnout_level = BurnoutLevel(label_id)
        except ValueError:
            burnout_level = BurnoutLevel.HEALTHY
        
        # Get recommendations for this level
        recs = self.recommendations.get(burnout_level, [])
        
        # Sort by priority
        sorted_recs = sorted(recs, key=lambda x: x.priority)
        
        # Format recommendations
        formatted_recs = []
        for rec in sorted_recs:
            formatted_recs.append({
                "category": rec.category,
                "title": rec.title,
                "description": rec.description,
                "priority": rec.priority,
                "action_items": rec.action_items
            })
        
        return {
            "burnout_level": burnout_level.name,
            "severity_score": self._calculate_severity(label_id, confidence),
            "summary": self._get_summary(burnout_level, confidence),
            "recommendations": formatted_recs,
            "emergency_resources": self._get_emergency_resources() if label_id == 2 else None,
            "follow_up": self._get_follow_up(burnout_level)
        }
    
    def _calculate_severity(self, label_id: int, confidence: float) -> float:
        """Calculate a severity score from 0-10"""
        base_score = label_id * 3.3  # 0, 3.3, or 6.6
        confidence_adjustment = confidence * 3.4 * (label_id / 2 if label_id > 0 else 0.3)
        return min(round(base_score + confidence_adjustment, 1), 10)
    
    def _get_summary(self, level: BurnoutLevel, confidence: float) -> str:
        """Generate a summary message based on burnout level"""
        summaries = {
            BurnoutLevel.HEALTHY: [
                "Great news! Your journal entry indicates a healthy mental state. Keep up the positive habits!",
                "You seem to be managing well. Your balanced approach is working - continue taking care of yourself!",
                "Your mental wellness indicators look positive. This is a good time to build resilience."
            ],
            BurnoutLevel.STRESSED: [
                "Your entry shows signs of academic stress. This is common and manageable with the right strategies.",
                "I'm noticing some stress indicators. Taking proactive steps now can prevent things from escalating.",
                "You're experiencing elevated stress levels. Let's work on some strategies to help you cope better."
            ],
            BurnoutLevel.BURNOUT: [
                "I'm concerned about what you've shared. Your entry suggests significant burnout. Please prioritize getting help.",
                "Your responses indicate you may be experiencing burnout. This is serious, but recovery is possible with support.",
                "I hear that you're struggling. Burnout is real and valid. Please reach out for professional support today."
            ]
        }
        
        return random.choice(summaries.get(level, summaries[BurnoutLevel.HEALTHY]))
    
    def _get_emergency_resources(self) -> Dict:
        """Get emergency mental health resources"""
        return {
            "crisis_lines": [
                {"name": "National Suicide Prevention Lifeline", "number": "988"},
                {"name": "Crisis Text Line", "text": "Text HOME to 741741"},
                {"name": "SAMHSA National Helpline", "number": "1-800-662-4357"}
            ],
            "campus_resources": [
                "Campus Counseling Center",
                "Student Health Services",
                "Dean of Students Office",
                "Residence Hall Advisors"
            ],
            "message": "If you're having thoughts of self-harm, please reach out immediately. You matter, and help is available 24/7."
        }
    
    def _get_follow_up(self, level: BurnoutLevel) -> str:
        """Get follow-up recommendation"""
        follow_ups = {
            BurnoutLevel.HEALTHY: "Consider journaling again in 1-2 weeks to track your wellbeing.",
            BurnoutLevel.STRESSED: "I recommend journaling daily this week and checking in again in 3-5 days.",
            BurnoutLevel.BURNOUT: "Please check in again tomorrow, and consider daily journaling as part of your recovery."
        }
        return follow_ups.get(level, follow_ups[BurnoutLevel.HEALTHY])
    
    def get_quick_tip(self, label_id: int) -> str:
        """Get a random quick tip based on burnout level"""
        tips = {
            0: [
                "💡 Tip: Share your positive strategies with a classmate who might be struggling!",
                "💡 Tip: This is a great time to start a new healthy habit!",
                "💡 Tip: Consider keeping a gratitude journal to maintain your positive mindset."
            ],
            1: [
                "💡 Tip: Take 5 deep breaths right now. Inhale for 4 counts, hold for 4, exhale for 6.",
                "💡 Tip: Step outside for 5 minutes. Fresh air can reset your mind.",
                "💡 Tip: Text a friend right now. Connection helps reduce stress."
            ],
            2: [
                "💡 Tip: You don't have to figure this out alone. Reach out to someone today.",
                "💡 Tip: One small step counts. Just focus on the next hour.",
                "💡 Tip: Your worth is not defined by your grades or productivity."
            ]
        }
        return random.choice(tips.get(label_id, tips[0]))


# Global advisor instance
_advisor = None


def get_advisor() -> BurnoutAdvisor:
    """Get or create the global advisor instance"""
    global _advisor
    if _advisor is None:
        _advisor = BurnoutAdvisor()
    return _advisor


def get_advice(prediction: Dict) -> Dict:
    """
    Convenience function to get advice from a prediction
    
    Args:
        prediction: Prediction dictionary from predict.py
        
    Returns:
        Advice dictionary with recommendations
    """
    advisor = get_advisor()
    return advisor.get_recommendations(prediction)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 BURNOUT ADVISOR - TEST MODE")
    print("="*60 + "\n")
    
    advisor = BurnoutAdvisor()
    
    # Test with different prediction scenarios
    test_predictions = [
        {"label": "Healthy", "label_id": 0, "confidence": 0.95},
        {"label": "Stressed", "label_id": 1, "confidence": 0.78},
        {"label": "Burnout", "label_id": 2, "confidence": 0.89}
    ]
    
    for pred in test_predictions:
        print(f"\n{'='*60}")
        print(f"📊 Prediction: {pred['label']} (Confidence: {pred['confidence']:.0%})")
        print("="*60)
        
        advice = advisor.get_recommendations(pred)
        
        print(f"\n📝 Summary: {advice['summary']}")
        print(f"📈 Severity Score: {advice['severity_score']}/10")
        print(f"📅 Follow-up: {advice['follow_up']}")
        
        print(f"\n💡 Quick Tip: {advisor.get_quick_tip(pred['label_id'])}")
        
        print("\n📋 Recommendations:")
        for i, rec in enumerate(advice['recommendations'], 1):
            print(f"\n   {rec['category']}: {rec['title']}")
            print(f"   {rec['description']}")
            print("   Action Items:")
            for item in rec['action_items'][:2]:  # Show first 2 items
                print(f"     • {item}")
        
        if advice.get('emergency_resources'):
            print("\n🚨 Emergency Resources Available")
