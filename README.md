# TC AI Prediction Tool

## Project Structure

TC AI Prediction Tool/
├── main.py                     # Starting point-launches the application
├── requirements.txt            # List of all software dependencies
├── pytest.ini                 # Testing configuration
│
├── core/                       # The "brain" of the application
│   ├── config.py              #  All application settings and configurations
│   ├── data_loader.py         #  Safely loads and validates your data files
│   ├── preprocess.py          #  Cleans and prepares data for AI models
│   ├── model.py               #  The AI magic-trains and evaluates models
│   ├── errors.py              #  Custom error handling for better user experience
│   ├── validators.py          #  Ensures your data is safe and valid
│   ├── database_utils.py      #  Manages database connections for logging
│   ├── logging.py             #  Records what happens for debugging
│   ├── logging_setup.py       #  Sets up the logging system
│   ├── async_operations.py    #  Handles background tasks
│   ├── model_worker.py        #  Background worker for model training
│   ├── resource_path.py       #  Finds files and resources
│   └── secure_config.py       #  Secure configuration management
│
├── ui/                         #  The visual interface you interact with
│   ├── app.py                 #  Main application window and all the buttons
│   ├── styles.py              #  Makes everything look pretty
│   └── dialogs.py             #  Pop-up windows and user interactions
│
├── tests/                      #  Quality assurance and testing
│   └── test_data_loader.py    # Tests to ensure data loading works correctly
│
├── Images/                     # Visual assets
│   └── icon.png               # Application icon
│
└── Font/                       # Custom fonts for the interface
    ├── Inter/                 # Modern, readable font family
    └── Caveat/                # Stylish accent font

## Key Components Explained

### The Core Engine (`core/` directory)

This is where all the smart stuff happens:
#### ** `model.py` - The AI Brain**
- Contains multiple machine learning algorithms (Logistic Regression, Random Forest, etc.)
- Automatically tests different approaches to find the best one for your data
- Handles both classification (predicting categories) and regression (predicting numbers)
- Uses cross-validation to ensure results are reliable

#### ** `preprocess.py` - The Data Cleaner**
- Takes messy, real-world data and makes it AI-ready
- Automatically detects what type of data you have (numbers, categories, text)
- Fills in missing values intelligently
- Converts text categories into numbers that AI can understand
- Scales everything to work well together

#### ** `data_loader.py` - The Safe File Handler**
- Securely loads Excel (.xlsx, .xls) and CSV files
- Protects against malicious files
- Checks file sizes to prevent crashes
- Detects file encoding automatically
- Validates data before processing

#### ** `config.py` - The Control Center**
- Contains all the settings that control how the application behaves
- Defines security limits (max file sizes, allowed file types)
- Sets machine learning parameters (how many models to try, etc.)
- Manages database connections for error logging

### The User Interface (`ui/` directory)

This is what you see and interact with:

#### ** `app.py` - The Main Application**
This is a comprehensive desktop application with:
- **File loading interface** - Drag and drop or browse for your data files
- **Data preview** - See your data in a table format
- **Column selection** - Choose which columns to use for predictions
- **Model training progress** - Watch as different AI models are tested
- **Results visualization** - Beautiful charts showing model performance
- **Prediction interface** - Make predictions on new data
- **Export functionality** - Save your trained models and results

#### ** `styles.py` - The Visual Designer**
- Defines the color scheme, fonts, and layout
- Makes buttons, tables, and dialogs look professional
- Ensures consistent appearance across the application

### Security & Reliability

#### ** Security Features**
- **File validation** - Only allows safe file types (.csv, .xlsx, .xls)
- **Size limits** - Prevents memory issues with huge files
- **Input sanitization** - Protects against malicious data
- **Error handling** - Graceful handling of problems

#### ** Logging & Monitoring**
- **Comprehensive logging** - Records everything that happens
- **Error tracking** - Captures and reports issues
- **Performance monitoring** - Tracks how well the application runs
- **Database integration** - Can store logs in a Neon database for analysis

## How It Works (Step by Step) 

1. **You start the application** (`main.py` runs)
2. **The system initializes** (sets up logging, checks database connection)
3. **You load your data file** (Excel or CSV)
4. **The app validates and previews your data**
5. **You select which column you want to predict** (the target)
6. **You choose which other columns to use** (the features)
7. **The AI automatically cleans your data** (handles missing values, converts types)
8. **Multiple AI models are trained and tested** (finds the best approach)
9. **You see results and charts** (understand how well each model performs)
10. **You can make predictions** (use the best model on new data)
11. **Everything is logged** (for debugging and improvement)

## Technology Stack 

This application is built with:

- **Python** - The main programming language
- **PyQt5** - Creates the desktop interface
- **pandas** - Handles data manipulation
- **scikit-learn** - Provides the machine learning algorithms
- **matplotlib** - Creates charts and visualizations
- **PostgreSQL/Neon** - Database for logging (optional)

## Machine Learning Models Included 

The tool automatically tries these proven algorithms:

### For Classification (Predicting Categories)
- **Logistic Regression** - Simple, fast, good for linear relationships
- **Random Forest** - Robust, handles complex patterns, good with mixed data types
- **Naive Bayes** - Great for text and categorical data
- **K-Nearest Neighbors** - Simple, good for local patterns

### For Regression (Predicting Numbers)
- **Linear Regression** - Simple, interpretable
- **Ridge Regression** - Prevents overfitting
- **Random Forest Regressor** - Handles complex, non-linear relationships
- **K-Nearest Neighbors Regressor** - Local pattern recognition

## Getting Started 

### Prerequisites
- Python 3.7 or higher
- All dependencies listed in `requirements.txt`

### Installation
1. **Install dependencies:**
   pip install -r requirements.txt

2. **Set up environment (optional):**
   - Create a `.env` file in the directory
   - Add `NEON_CONN_STR=your_database_connection_string` for error logging

3. **Run the application:**
   python main.py

### First Use
1. **Load your data** - Click "Browse" and select a CSV or Excel file
2. **Preview your data** - Check that it loaded correctly
3. **Select target column** - Choose what you want to predict
4. **Select feature columns** - Choose which data to use for predictions
5. **Train models** - Click "Train" and watch the AI work
6. **View results** - See which model performed best
7. **Make predictions** - Use your trained model on new data

## Configuration Options 

The application can be customized through `core/config.py`:

### Security Settings
- `MAX_FILE_SIZE_MB` - Maximum file size (default: 50MB)
- `ALLOWED_FILE_EXTENSIONS` - Permitted file types
- `CONNECTION_TIMEOUT` - Database connection timeout

### Machine Learning Settings
- `TEST_SIZE` - Portion of data used for testing (default: 20%)
- `CV_FOLDS` - Cross-validation folds (default: 3)
- `MAX_ROWS` - Maximum dataset size (default: 100,000 rows)
- `RANDOM_STATE` - Ensures reproducible results

### User Interface Settings
- `WINDOW_TITLE` - Application title
- `DEFAULT_FONT_SIZE` - Text size
- `DEFAULT_WINDOW_WIDTH/HEIGHT` - Initial window size

## Error Handling 

The application includes robust error handling:

- **Data Loading Errors** - Issues with file reading or parsing
- **Preprocessing Errors** - Problems cleaning or preparing data
- **Model Training Errors** - AI algorithm failures
- **Security Errors** - Attempts to load unsafe files
- **Validation Errors** - Data that doesn't meet requirements

All errors are logged and displayed in user-friendly language.

## Testing 

The application includes automated tests:


# Run all tests
pytest test/tests/

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

## Logging and Monitoring 

The application creates detailed logs:

- **Console logs** - Real-time feedback during use
- **File logs** - Persistent records of all activities
- **JSON logs** - Structured data for analysis
- **Database logs** - Optional centralized error tracking

Log files are stored in a writable directory and include:
- User actions
- System events
- Error details
- Performance metrics

## Contributing 

This codebase is well-structured for contributions:

1. **Core logic** is separated from UI
2. **Configuration** is centralized and flexible
3. **Error handling** is comprehensive
4. **Tests** ensure quality
5. **Logging** aids debugging

## Support 

If you encounter issues:

1. Check the log files for detailed error information
2. Ensure your data file meets the requirements
3. Verify all dependencies are installed correctly
4. Review the configuration settings

## Future Enhancements 

Potential improvements:
- Additional machine learning algorithms
- Advanced data visualization
- Real-time prediction API

