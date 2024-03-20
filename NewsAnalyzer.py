# DeepQ News Clickbait Tester
import gym
import numpy as np
import requests
from bs4 import BeautifulSoup
import spacy
from keras.layers import Input, Dense
from keras.models import Model
from collections import deque
import random

# Fetch articles from skynews using requests library
def fetch_articles(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML content
        articles = []
        for article_link in soup.find_all('a', href=True):
            article_url = article_link['href']
            if is_article_url(article_url):
                article_title = article_link.get_text().strip()
                if article_url.startswith("http"):  # Check if the URL is complete
                    article_content = extract_article_content(article_url)
                    if article_content is not None:
                        articles.append({'title': article_title, 'url': article_url, 'content': article_content})
                else:  # If URL is relative, prepend base URL
                    base_url = url if url.endswith("/") else url + "/"
                    article_url = base_url + article_url.lstrip("/")
                    article_content = extract_article_content(article_url)
                    if article_content is not None:
                        articles.append({'title': article_title, 'url': article_url, 'content': article_content})
        return articles
    else:
        print("Failed to fetch articles. Status code:", response.status_code)
        return []

def is_article_url(url):
    return "/video/" in url or "/story/" in url

def extract_article_content(url):
    try:
        if not url.startswith('http'):
            url = 'https://news.sky.com/' + url  # Handle relative URLs
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the content of the article from the webpage
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return content
        else:
            print(f"Failed to fetch article content from {url}. Status code:", response.status_code)
    except Exception as e:
        print(f"Failed to fetch article content from {url}. Error:", e)
    return None

# Define the ML environment
class NewsEnv(gym.Env):
    def __init__(self, articles):
        super(NewsEnv, self).__init__()
        self.articles = articles
        self.current_article = 0
        self.action_space = gym.spaces.Discrete(2)  # Action space: 0 = skip, 1 = read
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(articles),))  # Adjust observation space shape

    def reset(self):
        self.current_article = 0
        return self._get_state()

    def step(self, action):
        reward = self._get_reward(action)
        self.current_article += 1
        done = self.current_article >= len(self.articles)
        if not done:
            state = self._get_state()  # Update the state
        else:
            state = np.zeros(len(self.articles))  # Reset state if done
        return state, reward, done, {}

    def _get_state(self):
        # State representation: indicating whether each article has been read
        state = np.zeros(len(self.articles))
        state[:self.current_article + 1] = 1  # Set all articles before the current one as read
        return state

    def _get_reward(self, action):
        # Define reward function based on clickbait labels
        article = self.articles[self.current_article]
        title = article['title']
        content = article['content']
        
        # Analyze title and content using spaCy
        similarity_score = compute_similarity(title, content)
        
        if action == 1:
            return similarity_score  # Reward is similarity score
        else:
            return 0  # No reward if the article is skipped

def compute_similarity(text1, text2):
    # Load spaCy
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# Define the DQN model
def build_dqn_model(input_shape, action_space_size):
    input_layer = Input(shape=input_shape)
    dense1 = Dense(64, activation='relu')(input_layer)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(action_space_size, activation='linear')(dense2)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Define the experience replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=32, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = build_dqn_model((state_size,), action_size)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state.reshape(1, self.state_size))  # Reshape state
            return np.argmax(q_values[0])

    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        targets = self.model.predict_on_batch(np.array(states))
        next_state_targets = self.model.predict_on_batch(np.array(next_states))
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_state_targets[i])
            targets[i][actions[i]] = target
        self.model.train_on_batch(np.array(states), targets)

# Fetch articles from a hypothetical news website
url = "https://news.sky.com/" 
articles = fetch_articles(url)

if articles:  # Check if articles list is not empty
    print("Article analysis results:")
    for i, article in enumerate(articles, 1):
        title = article['title']
        url = article['url']
        env = NewsEnv([{'title': title, 'content': article['content']}])  # Create environment with a single article
        state = env.reset()
        action = 1  # Assume action is to read the article
        score = env._get_reward(action)
        print(f"Article {i}: {url} - Similarity Score: {score:.2f}")
else:
    print("No articles fetched. Exiting...")

# Corrected input shape based on the number of articles fetched
state_size = len(articles)

# Instantiate the DQN agent
state_size = len(articles)
action_size = 2  # Skip or read
agent = DQNAgent(state_size, action_size)

# Define the environment outside the loop
env = NewsEnv(articles)

# Training loop
num_episodes = 50  # Define the number of episodes for training
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.train()
