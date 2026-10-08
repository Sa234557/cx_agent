"""
Knowledge base for the FAQ agent.
In a real system this would be a database or document store.
Here we use a simple list of Q&A pairs that get loaded into ChromaDB.
"""

FAQ_DATA = [
    {
        "question": "How do I reset my password?",
        "answer": "To reset your password, click 'Forgot Password' on the login page. You will receive an email with a reset link within 5 minutes. The link expires in 24 hours."
    },
    {
        "question": "How do I cancel my subscription?",
        "answer": "You can cancel your subscription anytime from Settings > Billing > Cancel Subscription. Your access continues until the end of the billing period. No refunds are issued for partial months."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, and PayPal. Crypto payments are available for annual plans only."
    },
    {
        "question": "How do I contact support?",
        "answer": "You can contact support via live chat (available 24/7), email at support@example.com, or phone at 1-800-EXAMPLE during business hours (9AM-6PM EST)."
    },
    {
        "question": "What is your refund policy?",
        "answer": "We offer a 30-day money-back guarantee for new subscriptions. To request a refund, contact support within 30 days of your purchase with your order ID."
    },
    {
        "question": "How do I upgrade my plan?",
        "answer": "Go to Settings > Billing > Change Plan. Upgrades take effect immediately and you are billed the prorated difference. Downgrades take effect at the next billing cycle."
    },
    {
        "question": "Is my data secure?",
        "answer": "Yes. We use AES-256 encryption for data at rest and TLS 1.3 for data in transit. We are SOC 2 Type II certified and GDPR compliant. We never sell your data."
    },
    {
        "question": "How do I export my data?",
        "answer": "Go to Settings > Privacy > Export Data. You will receive a download link via email within 24 hours containing all your data in JSON format."
    },
    {
        "question": "Why was my account suspended?",
        "answer": "Accounts are suspended for violation of terms of service, suspicious activity, or failed payments. Check your email for details. Contact support to appeal a suspension."
    },
    {
        "question": "How do I add a team member?",
        "answer": "Go to Settings > Team > Invite Member. Enter their email and select their role (Admin, Editor, or Viewer). They will receive an invite email valid for 48 hours."
    }
]
