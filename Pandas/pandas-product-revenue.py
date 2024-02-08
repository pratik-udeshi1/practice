import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_sample_data():
    """Generate sample sales data for three products in three regions over a year."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-05-31', freq='D')
    products = ['Product A', 'Product B', 'Product C']
    regions = ['North', 'South', 'East']

    data = {
        'Date': np.random.choice(dates, 500),
        'Product': np.random.choice(products, 500),
        'Region': np.random.choice(regions, 500),
        'Sales': np.random.randint(10, 100, 500),
        'Revenue': np.random.uniform(100, 1000, 500)
    }

    return pd.DataFrame(data)


def preprocess_data(df):
    """Perform data preprocessing tasks."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    return df


def aggregate_data(df):
    """Group by month and region to get total sales and revenue."""
    return df.groupby(['Month', 'Region']).agg({'Sales': 'sum', 'Revenue': 'sum'}).reset_index()


def visualize_data(monthly_stats, regions):
    """Visualize the monthly sales trend for each region."""
    plt.figure(figsize=(12, 6))
    for region in regions:
        region_data = monthly_stats[monthly_stats['Region'] == region]
        plt.plot(region_data['Month'], region_data['Sales'], label=region)

    plt.title('Monthly Sales by Region')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Step 1: Generate sample data
    df = generate_sample_data()

    # Step 2: Data preprocessing
    df = preprocess_data(df)

    # Step 3: Aggregate data to get monthly total sales and revenue for each region
    monthly_stats = aggregate_data(df)

    # Step 4: Visualize the monthly sales trend for each region
    visualize_data(monthly_stats, regions=['North', 'South', 'East'])
