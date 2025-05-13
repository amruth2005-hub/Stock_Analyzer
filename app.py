from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import secrets
import re
import os

# Data and analysis libraries
import requests
import yfinance as yf
import wikipedia
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# AI and ML - Add Gemini integration
from google import genai

# Initialize Flask extensions
db = SQLAlchemy()
login_manager = LoginManager()
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Load API keys and ML models
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_gemini_apikey")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "your_alpha_vantage_apikey")

# Initialize Google Gemini client if API key is available
try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    gemini_available = True
except:
    gemini_available = False
    print("WARNING: Gemini API not available. Competitor analysis will be limited.")

# Initialize ML models if available
try:
    prediction_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ml_model_available = True
    print("Successfully loaded the prediction model")
except Exception as e:
    prediction_model = None
    scaler = None
    ml_model_available = False
    print(f"WARNING: ML model not available: {e}. Price prediction will be estimated.")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", 'sqlite:///stockmind_ai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_SECRET_KEY", secrets.token_hex(32))

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

# Validation functions
def validate_email(email):
    return bool(EMAIL_REGEX.match(email))

def validate_password(password):
    return len(password) >= 8

def validate_username(username):
    return len(username) >= 3

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    watchlists = db.relationship('Watchlist', backref='user', lazy=True)
    analyses = db.relationship('StockAnalysis', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Watchlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    stocks = db.relationship('WatchlistStock', backref='watchlist', lazy=True, cascade="all, delete-orphan")

class WatchlistStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    watchlist_id = db.Column(db.Integer, db.ForeignKey('watchlist.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

class StockAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    company_name = db.Column(db.String(100))
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    prediction_price = db.Column(db.Float)
    actual_price = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class PriceAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    target_price = db.Column(db.Float, nullable=False)
    alert_type = db.Column(db.String(10), nullable=False)  # 'above' or 'below'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    triggered = db.Column(db.Boolean, default=False)
    
    user = db.relationship('User', backref='alerts')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# JWT token required decorator for API routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
            
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
            
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
        
    return decorated

# Stock Analysis Functions

def create_lag_features(df, n_lags=5):
    """Create lag features for ML prediction model"""
    df_copy = df.copy()
    for i in range(1, n_lags + 1):
        df_copy[f'Close_Lag_{i}'] = df_copy['Close'].shift(i)
    df_copy['Volatility'] = df_copy['Close'].rolling(window=5).std()
    df_copy['Momentum'] = df_copy['Close'].pct_change(periods=5)
    df_copy['MA10_Ratio'] = df_copy['Close'] / df_copy['Close'].rolling(window=10).mean()
    df_copy['Volume_Change'] = df_copy['Volume'].pct_change()
    df_copy = df_copy.dropna()
    return df_copy

def fetch_wikipedia_summary(company_name): 
    """Fetch company summary from Wikipedia"""
    try: 
        search_results = wikipedia.search(company_name) 
        if search_results: 
            page_title = search_results[0] 
            summary = wikipedia.summary(page_title, sentences=3) 
            return page_title, summary 
    except Exception as e: 
        return None, f"Error fetching Wikipedia summary: {str(e)}" 
    return None, "No Wikipedia page found for the given company." 
 
def fetch_stock_price(ticker): 
    """Fetch stock price and historical data"""
    try: 
        stock = yf.Ticker(ticker) 
        history = stock.history(period="3mo") 
        time_labels = history.index.strftime('%Y-%m-%d').tolist() 
        stock_prices = [round(price, 2) for price in history['Close'].tolist()]
        return stock_prices, time_labels 
    except Exception as e: 
        return None, None

def get_ticker_from_alpha_vantage(company_name): 
    """Get ticker symbol from company name using Alpha Vantage"""
    try: 
        url = "https://www.alphavantage.co/query" 
        params = { 
            "function": "SYMBOL_SEARCH", 
            "keywords": company_name, 
            "apikey": ALPHA_VANTAGE_API_KEY, 
        } 
        response = requests.get(url, params=params) 
        data = response.json() 
        if "bestMatches" in data: 
            for match in data["bestMatches"]: 
                if match["4. region"] == "United States": 
                    return match["1. symbol"] 
        return None 
    except Exception as e: 
        return None 

def fallback_get_ticker(company_name):
    """Fallback method to get ticker if Alpha Vantage fails"""
    common_tickers = {
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'amazon': 'AMZN',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'meta': 'META',
        'facebook': 'META',
        'tesla': 'TSLA',
        'netflix': 'NFLX',
        'walmart': 'WMT',
    }
    
    # Check if the company name matches any in our mapping
    company_lower = company_name.lower()
    for name, ticker in common_tickers.items():
        if name in company_lower:
            return ticker
    
    # Try yfinance search as last resort
    try:
        ticker = yf.Ticker(company_name)
        if hasattr(ticker, 'info') and 'symbol' in ticker.info:
            return ticker.info['symbol']
    except:
        pass
    
    return None

def fetch_market_cap(ticker): 
    """Fetch market cap for a stock"""
    try: 
        stock = yf.Ticker(ticker) 
        market_cap = stock.info.get('marketCap', None) 
        return market_cap 
    except Exception as e: 
        return None 

def predict_stock_price(ticker):
    """Predict next day's stock price using ML model"""
    if not ml_model_available:
        # Fallback prediction if model isn't available
        # Get current price and estimate a simple % change
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="5d")
            current_price = history['Close'].iloc[-1]
            
            # Calculate average daily % change over last 5 days
            pct_changes = history['Close'].pct_change().dropna()
            avg_change = pct_changes.mean()
            
            # Predict tomorrow with average change
            predicted_price = round(current_price * (1 + avg_change), 2)
            return predicted_price, current_price
        except Exception as e:
            return None, None
    
    # Use ML model for prediction
    try:
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            return None, None
        
        current_price = round(df['Close'].iloc[-1], 2)
        
        # Process data for prediction
        df_processed = create_lag_features(df, n_lags=5)
        features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                   'Volatility', 'Momentum', 'MA10_Ratio', 'Volume_Change']
        
        if len(df_processed) > 0:
            X = df_processed[features]
            X_scaled = scaler.transform(X.tail(1))
            predicted_price = round(float(prediction_model.predict(X_scaled)[0]), 2)
            return predicted_price, current_price
        else:
            return None, current_price
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def query_gemini_llm(description):
    """Query Gemini LLM for competitor analysis"""
    if not gemini_available:
        # Fallback if Gemini is not available
        return fallback_competitor_analysis(description)
    
    try:
        prompt = f""" 
        Provide a structured list of sectors and their competitors for the following company description: 
        {description[:500]} 
        Format: 
        Sector Name : 
            Competitor 1 
            Competitor 2 
            Competitor 3 
 
        Leave a line after each sector. Do not use bullet points. 
        """ 
        response = genai_client.models.generate_content( 
            model="gemini-1.5-flash", contents=prompt 
        ) 
        content = response.candidates[0].content.parts[0].text 
        sectors = [] 
        for line in content.split("\n\n"): 
            lines = line.strip().split("\n") 
            if len(lines) > 1: 
                sector_name = lines[0].strip() 
                competitors = [l.strip() for l in lines[1:]] 
                sectors.append({"name": sector_name, "competitors": competitors}) 
        return sectors 
    except Exception as e:
        print(f"Gemini error: {e}")
        return fallback_competitor_analysis(description)

def fallback_competitor_analysis(description):
    """Fallback competitor analysis if Gemini is unavailable"""
    # Simple keyword-based analysis
    tech_companies = ['Microsoft', 'Apple', 'Google', 'Meta', 'Amazon', 'Netflix', 'IBM', 'Oracle', 'Intel']
    auto_companies = ['Ford', 'GM', 'Tesla', 'Toyota', 'Honda', 'BMW', 'Mercedes', 'Volkswagen']
    retail_companies = ['Walmart', 'Target', 'Costco', 'Amazon', 'Kroger', 'Home Depot']
    finance_companies = ['JPMorgan Chase', 'Bank of America', 'Wells Fargo', 'Citigroup', 'Goldman Sachs']
    
    sectors = []
    desc_lower = description.lower()
    
    if any(keyword in desc_lower for keyword in ['software', 'technology', 'tech', 'app', 'digital']):
        sectors.append({"name": "Technology Sector:", "competitors": tech_companies[:3]})
    
    if any(keyword in desc_lower for keyword in ['car', 'vehicle', 'automotive', 'automobile']):
        sectors.append({"name": "Automotive Sector:", "competitors": auto_companies[:3]})
        
    if any(keyword in desc_lower for keyword in ['retail', 'store', 'shop', 'e-commerce']):
        sectors.append({"name": "Retail Sector:", "competitors": retail_companies[:3]})
    
    if any(keyword in desc_lower for keyword in ['bank', 'finance', 'investment', 'money']):
        sectors.append({"name": "Financial Sector:", "competitors": finance_companies[:3]})
    
    # Default if no sector is detected
    if not sectors:
        sectors.append({"name": "General Business Sector:", "competitors": ['Amazon', 'Microsoft', 'Google']})
    
    return sectors

def get_top_competitors(competitors):
    """Get top competitors with stock data"""
    competitor_data = [] 
    processed_tickers = set()  # Track processed tickers to avoid duplicates
 
    for competitor in set(competitors):  # Remove duplicate names 
        ticker = get_ticker_from_alpha_vantage(competitor) 
        if not ticker:
            ticker = fallback_get_ticker(competitor)
            
        if ticker and ticker not in processed_tickers: 
            market_cap = fetch_market_cap(ticker) 
            stock_prices, time_labels = fetch_stock_price(ticker) 
            if market_cap and stock_prices and time_labels: 
                competitor_data.append({ 
                    "name": competitor, 
                    "ticker": ticker, 
                    "market_cap": market_cap, 
                    "stock_prices": stock_prices, 
                    "time_labels": time_labels, 
                    "stock_price": stock_prices[-1] if stock_prices else 0, 
                }) 
                processed_tickers.add(ticker)
 
    # Sort competitors by market cap and return the top 3 
    top_competitors = sorted(competitor_data, key=lambda x: x["market_cap"], reverse=True)[:3] 
    return top_competitors

def get_sector_performance(sector_name):
    """Get sector ETF performance data"""
    # Map common sectors to their ETFs
    sector_etfs = {
        "technology": "XLK",
        "tech": "XLK",
        "financial": "XLF",
        "finance": "XLF",
        "healthcare": "XLV",
        "health": "XLV",
        "consumer": "XLY",
        "retail": "XRT",
        "energy": "XLE",
        "utilities": "XLU",
        "industrial": "XLI",
        "materials": "XLB",
        "real estate": "XLRE",
        "communication": "XLC"
    }
    
    # Find matching ETF
    sector_lower = sector_name.lower()
    etf_ticker = None
    
    for key, ticker in sector_etfs.items():
        if key in sector_lower:
            etf_ticker = ticker
            break
    
    # Default to SPY (S&P 500) if no match
    if not etf_ticker:
        etf_ticker = "SPY"
    
    # Get ETF data
    try:
        stock_prices, time_labels = fetch_stock_price(etf_ticker)
        if stock_prices and time_labels:
            return {
                "name": f"{sector_name} Sector ETF",
                "ticker": etf_ticker,
                "stock_prices": stock_prices,
                "time_labels": time_labels,
                "current_price": stock_prices[-1] if stock_prices else 0
            }
    except:
        pass
    
    return None

def get_company_news(ticker):
    """Get recent news for a company"""
    # Simple implementation using yfinance
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if news:
            # Format and limit news
            formatted_news = []
            for item in news[:5]:  # Limit to 5 most recent items
                formatted_news.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime('%Y-%m-%d %H:%M')
                })
            return formatted_news
    except:
        pass
    
    return []

def get_growth_metrics(ticker):
    """Get growth metrics for a company"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key growth metrics
        metrics = {
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "beta": info.get("beta")
        }
        
        # Format percentages and handle None values
        formatted_metrics = {}
        for key, value in metrics.items():
            if value is None:
                formatted_metrics[key] = "N/A"
            elif key in ["revenue_growth", "earnings_growth", "dividend_yield", "profit_margins", "return_on_equity"]:
                formatted_metrics[key] = f"{value:.2%}" if value else "N/A"
            else:
                formatted_metrics[key] = f"{value:.2f}" if value else "N/A"
                
        return formatted_metrics
    except:
        return {
            "revenue_growth": "N/A",
            "earnings_growth": "N/A",
            "dividend_yield": "N/A",
            "profit_margins": "N/A",
            "return_on_equity": "N/A",
            "beta": "N/A"
        }

# Routes

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Basic validation
        if not email or not password:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('login'))
            
        if not validate_email(email):
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('login'))
            
        user = User.query.filter_by(email=email).first()
        
        if user is None or not user.check_password(password):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
            
        login_user(user)
        flash('Logged in successfully!', 'success')
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard'))
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Server-side validation
        if not validate_username(username):
            flash('Username must be at least 3 characters', 'error')
            return redirect(url_for('register'))
            
        if not validate_email(email):
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('register'))
            
        if not validate_password(password):
            flash('Password must be at least 8 characters long', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'error')
            return redirect(url_for('register'))
            
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Create default watchlist for new user
        default_watchlist = Watchlist(name="My Watchlist", user_id=user.id)
        db.session.add(default_watchlist)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful!', 'success')
    return redirect(url_for('login'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    # Get user's watchlists and recently analyzed stocks
    watchlists = Watchlist.query.filter_by(user_id=current_user.id).all()
    recent_analyses = StockAnalysis.query.filter_by(user_id=current_user.id).order_by(StockAnalysis.analysis_date.desc()).limit(5).all()
    
    return render_template("dashboard.html", 
                          user=current_user, 
                          watchlists=watchlists,
                          recent_analyses=recent_analyses)

@app.route("/analyze")
@login_required
def analyze_view():
    """Render the analysis page"""
    return render_template("analyze.html")

@app.route("/portfolio")
@login_required
def portfolio():
    """Render the portfolio page"""
    watchlists = Watchlist.query.filter_by(user_id=current_user.id).all()
    return render_template("portfolio.html", watchlists=watchlists)

@app.route("/watchlist/<int:watchlist_id>")
@login_required
def view_watchlist(watchlist_id):
    """View a specific watchlist"""
    watchlist = Watchlist.query.get_or_404(watchlist_id)
    
    # Check if user owns this watchlist
    if watchlist.user_id != current_user.id:
        flash("You don't have permission to view this watchlist", "error")
        return redirect(url_for('dashboard'))
    
    # Get stock data for each stock in the watchlist
    stocks_data = []
    for stock in watchlist.stocks:
        try:
            ticker_data = yf.Ticker(stock.ticker)
            history = ticker_data.history(period="1d")
            current_price = round(history['Close'].iloc[-1], 2) if not history.empty else None
            
            # Get basic stock info
            info = ticker_data.info
            name = info.get('shortName', stock.ticker)
            
            stocks_data.append({
                'id': stock.id,
                'ticker': stock.ticker,
                'name': name,
                'price': current_price,
                'change': info.get('regularMarketChangePercent', None)
            })
        except:
            stocks_data.append({
                'id': stock.id,
                'ticker': stock.ticker,
                'name': stock.ticker,
                'price': 'N/A',
                'change': None
            })
    
    return render_template("watchlist.html", watchlist=watchlist, stocks=stocks_data)

# API Endpoints

@app.route("/api/auth/login", methods=['POST'])
def api_login():
    data = request.get_json()
    
    # Validate input
    if not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400
    
    if not validate_email(data['email']):
        return jsonify({'message': 'Please enter a valid email address'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'message': 'Invalid credentials'}), 401
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.config['JWT_SECRET_KEY'])
    
    return jsonify({
        'token': token,
        'userId': user.id,
        'email': user.email
    })

@app.route("/api/auth/register", methods=['POST'])
def api_register():
    data = request.get_json()
    
    # Validate input
    if not data.get('email') or not data.get('password') or not data.get('username'):
        return jsonify({'message': 'Username, email, and password are required'}), 400
    
    if not validate_email(data['email']):
        return jsonify({'message': 'Please enter a valid email address'}), 400
        
    if not validate_password(data['password']):
        return jsonify({'message': 'Password must be at least 8 characters'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already taken'}), 400
    
    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    # Create default watchlist
    default_watchlist = Watchlist(name="My Watchlist", user_id=user.id)
    db.session.add(default_watchlist)
    db.session.commit()
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }, app.config['JWT_SECRET_KEY'])
    
    return jsonify({
        'token': token,
        'userId': user.id,
        'email': user.email
    })

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze_company():
    """Analyze a company with complete data"""
    data = request.json
    company_name = data.get('company_name', '')
    ticker_input = data.get('ticker', '')
    
    # Either ticker or company name must be provided
    if not ticker_input and not company_name:
        return jsonify(success=False, error="Please provide either a company name or ticker symbol.")
    
    # If ticker is not provided, try to get it from company name
    if not ticker_input and company_name:
        ticker = get_ticker_from_alpha_vantage(company_name)
        if not ticker:
            ticker = fallback_get_ticker(company_name)
    else:
        ticker = ticker_input
    
    if not ticker:
        return jsonify(success=False, error="Could not determine ticker symbol. Please provide a valid ticker.")
    
    # Get company description
    if company_name:
        _, description = fetch_wikipedia_summary(company_name)
    else:
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            _, description = fetch_wikipedia_summary(company_name)
        except:
            description = f"No information available for {ticker}"
            company_name = ticker
    
    if not description or "Error fetching" in description:
        description = f"No detailed information available for {company_name}."
    
    # Fetch stock prices
    stock_prices, time_labels = fetch_stock_price(ticker)
    if not stock_prices or not time_labels:
        return jsonify(success=False, error="Could not fetch stock prices.")# Fetch stock prices
    stock_prices, time_labels = fetch_stock_price(ticker)
    if not stock_prices or not time_labels:
        return jsonify(success=False, error="Could not fetch stock prices.")
    
    # Calculate prediction
    predicted_price, current_price = predict_stock_price(ticker)
    
    # Get competitors from LLM
    sectors = query_gemini_llm(description)
    
    # Extract all competitors
    all_competitors = []
    for sector in sectors:
        all_competitors.extend(sector["competitors"])
    
    # Get top competitors
    top_competitors = get_top_competitors(all_competitors)
    
    # Get sector performance data
    sector_performances = []
    for sector in sectors:
        sector_name = sector["name"].replace(":", "").strip()
        sector_perf = get_sector_performance(sector_name)
        if sector_perf:
            sector_performances.append(sector_perf)
    
    # Get company news
    news = get_company_news(ticker)
    
    # Get growth metrics
    growth_metrics = get_growth_metrics(ticker)
    
    # Save the analysis
    new_analysis = StockAnalysis(
        ticker=ticker,
        company_name=company_name,
        prediction_price=predicted_price,
        actual_price=current_price,
        user_id=current_user.id
    )
    db.session.add(new_analysis)
    db.session.commit()
    
    # Return the full analysis
    return jsonify({
        'success': True,
        'ticker': ticker,
        'company_name': company_name,
        'description': description,
        'stock_prices': stock_prices,
        'time_labels': time_labels,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'sectors': sectors,
        'top_competitors': top_competitors,
        'sector_performances': sector_performances,
        'news': news,
        'growth_metrics': growth_metrics
    })

@app.route("/api/watchlist", methods=["GET"])
@login_required
def get_watchlists():
    """Get all watchlists for the current user"""
    watchlists = Watchlist.query.filter_by(user_id=current_user.id).all()
    
    result = []
    for watchlist in watchlists:
        stocks = [{'id': stock.id, 'ticker': stock.ticker} for stock in watchlist.stocks]
        result.append({
            'id': watchlist.id,
            'name': watchlist.name,
            'stocks': stocks
        })
    
    return jsonify(result)

@app.route("/api/watchlist", methods=["POST"])
@login_required
def create_watchlist():
    """Create a new watchlist"""
    data = request.json
    
    if not data.get('name'):
        return jsonify({'message': 'Watchlist name is required'}), 400
    
    watchlist = Watchlist(name=data['name'], user_id=current_user.id)
    db.session.add(watchlist)
    db.session.commit()
    
    return jsonify({
        'id': watchlist.id,
        'name': watchlist.name,
        'stocks': []
    })

@app.route("/api/watchlist/<int:watchlist_id>", methods=["DELETE"])
@login_required
def delete_watchlist(watchlist_id):
    """Delete a watchlist"""
    watchlist = Watchlist.query.get_or_404(watchlist_id)
    
    # Check if user owns this watchlist
    if watchlist.user_id != current_user.id:
        return jsonify({'message': "You don't have permission to delete this watchlist"}), 403
    
    db.session.delete(watchlist)
    db.session.commit()
    
    return jsonify({'message': 'Watchlist deleted successfully'})

@app.route("/api/watchlist/<int:watchlist_id>/stock", methods=["POST"])
@login_required
def add_stock_to_watchlist(watchlist_id):
    """Add a stock to a watchlist"""
    watchlist = Watchlist.query.get_or_404(watchlist_id)
    
    # Check if user owns this watchlist
    if watchlist.user_id != current_user.id:
        return jsonify({'message': "You don't have permission to modify this watchlist"}), 403
    
    data = request.json
    
    if not data.get('ticker'):
        return jsonify({'message': 'Ticker symbol is required'}), 400
    
    # Check if stock already exists in watchlist
    existing_stock = WatchlistStock.query.filter_by(
        watchlist_id=watchlist_id, 
        ticker=data['ticker'].upper()
    ).first()
    
    if existing_stock:
        return jsonify({'message': 'Stock already exists in watchlist'}), 400
    
    stock = WatchlistStock(ticker=data['ticker'].upper(), watchlist_id=watchlist_id)
    db.session.add(stock)
    db.session.commit()
    
    return jsonify({
        'id': stock.id,
        'ticker': stock.ticker,
        'watchlist_id': watchlist_id
    })

@app.route("/api/watchlist/stock/<int:stock_id>", methods=["DELETE"])
@login_required
def remove_stock_from_watchlist(stock_id):
    """Remove a stock from a watchlist"""
    stock = WatchlistStock.query.get_or_404(stock_id)
    
    # Check if user owns the watchlist that contains this stock
    watchlist = Watchlist.query.get(stock.watchlist_id)
    if watchlist.user_id != current_user.id:
        return jsonify({'message': "You don't have permission to remove this stock"}), 403
    
    db.session.delete(stock)
    db.session.commit()
    
    return jsonify({'message': 'Stock removed successfully'})

@app.route("/api/alerts", methods=["GET"])
@login_required
def get_alerts():
    """Get all price alerts for the current user"""
    alerts = PriceAlert.query.filter_by(user_id=current_user.id).all()
    
    result = []
    for alert in alerts:
        result.append({
            'id': alert.id,
            'ticker': alert.ticker,
            'target_price': alert.target_price,
            'alert_type': alert.alert_type,
            'triggered': alert.triggered,
            'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(result)

@app.route("/api/alerts", methods=["POST"])
@login_required
def create_alert():
    """Create a new price alert"""
    data = request.json
    
    if not data.get('ticker') or not data.get('target_price') or not data.get('alert_type'):
        return jsonify({'message': 'Ticker, target price, and alert type are required'}), 400
    
    if data['alert_type'] not in ['above', 'below']:
        return jsonify({'message': 'Alert type must be either "above" or "below"'}), 400
    
    try:
        target_price = float(data['target_price'])
    except:
        return jsonify({'message': 'Target price must be a valid number'}), 400
    
    alert = PriceAlert(
        ticker=data['ticker'].upper(),
        target_price=target_price,
        alert_type=data['alert_type'],
        user_id=current_user.id
    )
    db.session.add(alert)
    db.session.commit()
    
    return jsonify({
        'id': alert.id,
        'ticker': alert.ticker,
        'target_price': alert.target_price,
        'alert_type': alert.alert_type,
        'triggered': alert.triggered,
        'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
@login_required
def delete_alert(alert_id):
    """Delete a price alert"""
    alert = PriceAlert.query.get_or_404(alert_id)
    
    # Check if user owns this alert
    if alert.user_id != current_user.id:
        return jsonify({'message': "You don't have permission to delete this alert"}), 403
    
    db.session.delete(alert)
    db.session.commit()
    
    return jsonify({'message': 'Alert deleted successfully'})

@app.route("/api/recent-analyses", methods=["GET"])
@login_required
def get_recent_analyses():
    """Get recent stock analyses for the current user"""
    analyses = StockAnalysis.query.filter_by(user_id=current_user.id).order_by(StockAnalysis.analysis_date.desc()).limit(10).all()
    
    result = []
    for analysis in analyses:
        result.append({
            'id': analysis.id,
            'ticker': analysis.ticker,
            'company_name': analysis.company_name,
            'prediction_price': analysis.prediction_price,
            'actual_price': analysis.actual_price,
            'analysis_date': analysis.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(result)

# Keep the original analyze endpoint from your old app.py for backward compatibility
@app.route('/analyze', methods=["POST"])
def analyze():
    data = request.json
    company_name = data.get('company_name', '')
    ticker_input = data.get('ticker', '')
    source = data.get('source', 'yfinance')
    
    # If ticker is not provided, try to get it from company name
    if not ticker_input and company_name:
        ticker = get_ticker_from_alpha_vantage(company_name)
        if not ticker:
            ticker = fallback_get_ticker(company_name)
    else:
        ticker = ticker_input
    
    if not ticker:
        return jsonify({
            'success': False,
            'error': 'Could not determine ticker symbol. Please provide a valid ticker.'
        })
    
    # Get company description
    if company_name:
        description = fetch_wikipedia_summary(company_name)[1]
    else:
        try:
            # Try to get company name from ticker
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            description = fetch_wikipedia_summary(company_name)[1]
        except:
            description = f"Information for {ticker}"
            company_name = ticker
    
    try:
        # Fetch stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': f"No data found for ticker {ticker}"
            })
        
        # Get current price and historical data
        current_price = round(df['Close'].iloc[-1], 2)
        stock_prices = df['Close'].tolist()
        time_labels = df.index.strftime('%Y-%m-%d').tolist()
        
        # Use your Linear Regression model to predict next day's price
        predicted_price, _ = predict_stock_price(ticker)
        
        # Get competitors - enhanced from original app.py
        sectors = query_gemini_llm(description)
        all_competitors = []
        for sector in sectors:
            all_competitors.extend(sector["competitors"])
        top_competitors = get_top_competitors(all_competitors)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'company_name': company_name,
            'description': description,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'stock_prices': stock_prices,
            'time_labels': time_labels,
            'top_competitors': top_competitors,
            'sectors': sectors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        })

# Initialize the database
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)