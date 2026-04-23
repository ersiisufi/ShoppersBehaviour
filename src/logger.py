import logging
import os

def logger_setup(log_file='app.log'):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        filename=os.path.join('logs', log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("PurchaseIntent")

logger=logger_setup()