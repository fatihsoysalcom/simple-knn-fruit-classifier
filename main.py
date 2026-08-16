import math

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

def classify_fruit(training_data, new_fruit_features, k=3):
    """
    Classifies a new fruit using the K-Nearest Neighbors (KNN) algorithm.
    training_data: List of (features, label) tuples.
    new_fruit_features: Features of the fruit to classify.
    k: Number of nearest neighbors to consider.
    """
    distances = []
    # Calculate distance from new_fruit to every training example
    for features, label in training_data:
        dist = euclidean_distance(features, new_fruit_features)
        distances.append((dist, label)) # Store distance and corresponding label

    # Sort by distance and get the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    # Count votes for each label among the k neighbors
    label_votes = {}
    for _, label in k_nearest_neighbors:
        label_votes[label] = label_votes.get(label, 0) + 1

    # Return the label with the most votes
    # This simulates the "learning" and "prediction" aspect of ML
    predicted_label = max(label_votes, key=label_votes.get)
    return predicted_label

if __name__ == "__main__":
    # This represents our "training data" - examples the AI learns from.
    # Features: [weight (grams), smoothness (0=rough, 1=smooth)]
    # Labels: "Apple" or "Orange"
    training_data = [
        ([150, 0.8], "Apple"),   # Example 1: Apple
        ([160, 0.7], "Apple"),   # Example 2: Apple
        ([140, 0.9], "Apple"),   # Example 3: Apple
        ([180, 0.6], "Orange"),  # Example 4: Orange
        ([190, 0.5], "Orange"),  # Example 5: Orange
        ([170, 0.7], "Orange"),  # Example 6: Orange
        ([155, 0.85], "Apple"),  # Example 7: Apple
        ([175, 0.65], "Orange"), # Example 8: Orange
    ]

    print("--- Simple K-Nearest Neighbors (KNN) Classifier ---")
    print("Training Data (Weight, Smoothness) -> Fruit Type:")
    for features, label in training_data:
        print(f"  {features} -> {label}")
    print("\n")

    # New fruit to classify
    new_fruit1_features = [152, 0.82] # Looks like an apple
    new_fruit2_features = [185, 0.55] # Looks like an orange
    new_fruit3_features = [165, 0.75] # Borderline case

    # Make predictions using our simple "AI" model
    # This is where the AI applies what it "learned" from the training data.
    predicted_type1 = classify_fruit(training_data, new_fruit1_features, k=3)
    print(f"New fruit with features {new_fruit1_features}: Predicted type = {predicted_type1}")

    predicted_type2 = classify_fruit(training_data, new_fruit2_features, k=3)
    print(f"New fruit with features {new_fruit2_features}: Predicted type = {predicted_type2}")

    predicted_type3 = classify_fruit(training_data, new_fruit3_features, k=3)
    print(f"New fruit with features {new_fruit3_features}: Predicted type = {predicted_type3}")

    print("\nThis example demonstrates how a simple machine learning algorithm can 'learn' patterns from data and make predictions.")