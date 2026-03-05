import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from telegram.ext import CallbackQueryHandler

user_history = {}

# 1. Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# 2. Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# 3. Translation Logic
async def translate_text(user_input):
    # This new prompt tells the AI to decide which direction to translate
    prompt = ChatPromptTemplate.from_template(
        "You are an expert English-Amharic translator. "
        "Detect the language of the input. "
        "If the input is NOT English and NOT Amharic, reply ONLY with the phrase: 'UNSUPPORTED_LANGUAGE'. "
        "If it is English, translate it to Amharic. "
        "If it is Amharic, translate it to English. "
        "Provide only the translation, nothing else.\n\n"
        "Input: {text}"
    )
    chain = prompt | llm
    response = chain.invoke({"text": user_input})
    return response.content

# 4. START COMMAND HANDLER 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 **Selam! I'm Ge'ez**, your bidirectional translator.\n\n"
        "I can now translate in both directions:\n"
        "   🇬🇧 **English** ➡️ 🇪🇹 **Amharic**\n"
        "   🇪🇹 **Amharic** ➡️ 🇬🇧 **English**\n\n"
        "Just send me any word or sentence in either language!"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

# 5. Help Handler 
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 **How to use Ge'ez:**\n\n"
        "1️⃣ Send any English text, and I will translate it to Amharic or vice versa.\n"
        "2️⃣ Use /start to see the welcome message again.\n"
        "3️⃣ Use /translate if you want a quick reminder of how I work.\n\n"
        "I'm here to make English/Amharic easier to understand! 😊"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

# 6. Translate Handler 
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✨ **Quick Translation Mode**\n\n"
        "Go ahead! Type the English/Amharic word or sentence you're thinking of right now."
    )

# 7. History Handler
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # Get history for this specific user, or an empty list if they have none
    user_data = user_history.get(user_id, [])
    
    if not user_data:
        await update.message.reply_text("📜 Your history is empty! Start translating to fill it up.")
        return
    
    msg = "📜 **Your Recent Translations:**\n\n"
    for i, item in enumerate(user_data, 1):
        msg += f"{i}. {item}\n"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

# 7. Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_id = update.effective_user.id
    
    # 1. Check for simple greetings first
    if user_text.lower() in ["hi", "hello", "hey"]:
        await update.message.reply_text("Hello! I'm ready. Please send me any English or Amharic text you want translated! 😊")
        return

    # 2. Show "typing..." while Groq works
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # 3. Get translation FIRST (This defines 'translation_result')
    translation_result = await translate_text(user_text)
    
    # 4. NOW check if it's an unsupported language
    if "UNSUPPORTED_LANGUAGE" in translation_result:
        await update.message.reply_text(
            "⚠️ **Sorry!** I only understand English and Amharic for now.\n\nPlease try a sentence in one of those two languages! 😊",
            parse_mode='Markdown'
        )
        return

    # 5. LOGIC TO SAVE HISTORY 
    if user_id not in user_history:
        user_history[user_id] = []
    
    entry = f"{user_text[:20]}... ➡️ {translation_result[:20]}..."
    user_history[user_id].append(entry)
    
    if len(user_history[user_id]) > 3:
        user_history[user_id].pop(0)

    # 6. INTERACTIVE BUTTONS
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup # Ensure these are imported
    keyboard = [
        [
            InlineKeyboardButton("🗑️ Clear History", callback_data="clear_history")
        ],
        [InlineKeyboardButton("📢 Share Translation", switch_inline_query=translation_result)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # 7. Final Reply (Tap-to-Copy formatting)
    await update.message.reply_text(
        f"✨ **Translation:**\n`{translation_result}`", 
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    await query.answer() 
    
    if query.data == "clear_history":
        user_history[user_id] = []
        await query.edit_message_text("🗑️ Your translation history has been cleared.")

# 8. Unknown Command Handler
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤔 Sorry, I don't recognize that command. Use /help to see what I can do!")

# 9. Start the Bot
if __name__ == '__main__':
    print("Ge'ez Bot is LIVE...")
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Register commands in order
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    # Register text handler (all non-command text)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Register unknown command handler (ANY slash command not listed above)
    app.add_handler(MessageHandler(filters.COMMAND, unknown))
    
    app.run_polling()