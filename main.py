import os
import csv
from twitchio import Client, Message
from datetime import datetime

# Set up your Twitch OAuth token and Channel name
TOKEN = 'xxxxxxxxxxxxxxxxx'  # Replace with your OAuth token
CLIENT_ID = 'xxxxxxxxxxxxxxxxxx'  # Replace with your Twitch client ID
CHANNEL = 'Mr_Mammal'  # Replace with the channel you want to connect to
# CSV file path for storing chat logs
CSV_FILE = 'twitch-chat-log.csv'


# Function to write the username and message to CSV file
def write_to_csv(username, message_content):
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([username, message_content])


# Define the client for Twitch
class BotClient(Client):
    def __init__(self):
        super().__init__(token=TOKEN, initial_channels=[CHANNEL])

    async def event_ready(self):
        print(f'Bot connected to Twitch! Listening to chat in {CHANNEL}.')

        # Write header to CSV file if the file is empty (first run)
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Username', 'Message'])  # Write column headers

    async def event_message(self, message: Message):
        # Write the username and message content to the CSV file
        write_to_csv(message.author.name, message.content)

        # Optionally, print the message content
        print(f'{message.author.name}: {message.content}')


# Start the bot
bot = BotClient()
bot.run()
