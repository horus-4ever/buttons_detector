from .datasetvisualizer import *


if __name__ == "__main__":
    dataset_path = Path("sim_p1.dataset copy.json")
    dataset = DatasetConfig.open(config_path=dataset_path).load()
    # training dataset is representative of the dataset so let's use it
    train_dataset = dataset.train_annotations
    plot_counterpart_kmeans(train_dataset)
    plot_buttons_distribution(train_dataset)
    plot_counterpart_size(train_dataset)
    counterpart_stats = get_counterpart_size_stats(train_dataset)
    print("counterpart Size Statistics:")
    for dimension, stats in counterpart_stats.items():
        print(f"  {dimension.capitalize()}:")
        for stat, value in stats.items():
            print(f"    {stat.capitalize()}: {value}")

    button_stats = get_button_size_stats(train_dataset)
    print("\nButton Size Statistics:")
    for dimension, stats in button_stats.items():
        print(f"  {dimension.capitalize()}:")
        for stat, value in stats.items():
            print(f"    {stat.capitalize()}: {value}")