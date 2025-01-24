import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def db_statistics(df):
    # ----------- Summary Statistics -----------
    total_people = len(df)
    total_images = df['images'].sum()
    avg_images_per_person = df['images'].mean()
    median_images_per_person = df['images'].median()
    max_images = df['images'].max()
    min_images = df['images'].min()
    single_count = (df['images'] == 1).sum()
    multi_count = (df['images'] > 1).sum()

    print(f"Total People: {total_people}")
    print(f"Total Images: {int(total_images)}")
    print(f"Max Images for a Person: {int(max_images)}")
    print(f"Total People with one Image: {single_count}")
    print(f"Total People with multiple Images: {multi_count}")

    # ----------- Top 10 People with Most Images -----------
    top_people = df.sort_values(by='images', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(y=top_people['name'], x=top_people['images'])
    plt.ylabel("Person Name")
    plt.title("Top 10 People with Most Images")
    plt.show()

    # ----------- Percentage of People with Only 1 Image -----------
    single_count = (df['images'] == 1).sum()
    multi_count = (df['images'] > 1).sum()
    labels = ['Single Image', 'Multiple Images']
    sizes = [single_count, multi_count]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
    plt.title("Proportion of People with Single vs. Multiple Images")
    plt.show() 

