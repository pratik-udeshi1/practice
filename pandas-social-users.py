import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_sample_data():
    """Generate sample user activity data for a social media platform over a month."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-01-31', freq='D')
    users = ['User_A', 'User_B', 'User_C']
    activities = ['Post', 'Like', 'Comment', 'Share']

    data = {
        'Date': np.random.choice(dates, 1000),
        'User': np.random.choice(users, 1000),
        'Activity': np.random.choice(activities, 1000),
        'Likes': np.random.randint(1, 100, 1000),
        'Comments': np.random.randint(1, 20, 1000)
    }

    return pd.DataFrame(data)


def preprocess_data(df):
    """Perform data preprocessing tasks."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.day_name()
    return df


def daily_activity_analysis(df):
    """Analyze daily activity count."""
    return df.groupby(['Date', 'Activity']).size().unstack(fill_value=0)


def weekly_user_engagement_analysis(df):
    """Analyze weekly user engagement."""
    return df.groupby(['Weekday', 'User']).agg({'Likes': 'sum', 'Comments': 'sum'}).reset_index()


def most_active_users_analysis(df):
    """Identify the most active users."""
    return df.groupby('User').agg({'Likes': 'sum', 'Comments': 'sum'}).sort_values(by='Likes', ascending=False)[:5]


# Main execution
if __name__ == "__main__":
    # Step 1: Generate sample data
    df = generate_sample_data()

    # Step 2: Data preprocessing
    df = preprocess_data(df)

    # Step 3: Daily activity analysis
    daily_activity_count = daily_activity_analysis(df)

    # Step 4: Weekly user engagement analysis
    weekly_user_engagement = weekly_user_engagement_analysis(df)

    # Step 5: Most active users analysis
    most_active_users = most_active_users_analysis(df)

    # Visualization 1: Daily activity trends
    plt.figure(figsize=(12, 6))
    for activity in df['Activity'].unique():
        plt.plot(daily_activity_count.index, daily_activity_count[activity], label=activity)

    plt.title('Daily Activity Trends')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # Visualization 2: Weekly user engagement bar chart
    weekly_user_engagement.plot(kind='bar', x='User', y=['Likes', 'Comments'], stacked=True)
    plt.title('Weekly User Engagement')
    plt.xlabel('User')
    plt.ylabel('Count')
    plt.show()

    most_active_users.plot(kind='bar', y=['Likes', 'Comments'], stacked=True, color=['seagreen', 'lightgreen'])
    plt.title('Most Active Users with Likes and Comments')
    plt.xlabel('User')
    plt.ylabel('Count')
    plt.show()

    # Print most active users
    print("Most Active Users:")
    print(most_active_users)
