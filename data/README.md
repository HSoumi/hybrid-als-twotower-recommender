# Dataset Information

## Amazon E-commerce Dataset

**Source**: https://github.com/aksharpandia/miniamazon_data  
**Original Size**: 10,000 products with 17 attributes  
**Format**: CSV file (`amazon_co-ecommerce_sample.csv`)  
**Usage**: Academic research on hybrid recommender systems

## Data Structure

The dataset contains 17 columns with Amazon e-commerce product information:

| Column Name | Description | Data Type | Usage in Research |
|-------------|-------------|-----------|-------------------|
| `uniq_id` | Unique identifier | Integer | Renamed to `userId` for collaborative filtering |
| `product_name` | Product title/name | String | Used to create `itemId` via label encoding |
| `manufacturer` | Product brand/manufacturer | String | Content-based feature (label encoded) |
| `price` | Product price in £ format | String | Content-based feature (label encoded) |
| `number_available_in_stock` | Stock quantity | String/Integer | Auxiliary feature |
| `number_of_reviews` | Total review count | Integer | Quality indicator |
| `number_of_answered_questions` | Q&A count | Integer | **Dropped** (ineffective for recommendations) |
| `average_review_rating` | Rating in "X out of 5 stars" format | String | **Primary target variable** (label encoded) |
| `amazon_category_and_sub_category` | Product category hierarchy | String | Content-based feature (label encoded) |
| `customers_who_bought_this_item_also_bought` | Related product information | String | Collaborative signal |
| `description` | Product description | String | Content-based feature |
| `product_information` | Technical specifications | String | Content-based feature |
| `product_description` | Marketing description | String | Content-based feature |
| `items_customers_buy_after_viewing_this_item` | Sequential behavior data | String | Behavioral signal |
| `customer_questions_and_answers` | Q&A content | String | **Dropped** (high missing values) |
| `customer_reviews` | Review text content | String | Content-based feature |
| `sellers` | Seller information | String | Marketplace feature |

## Sample Data Characteristics

### **Product Categories**
- **Domain**: Model trains and railway sets
- **Example category**: "Hobbies > Model Trains & Railway Sets > Rail Vehicles > Trains"
- **Manufacturers**: Hornby, FunkyBuys, Kato, Bachmann, PECO

### **Price Format**
- **Currency**: British Pounds (£)
- **Examples**: £3.42, £16.99, £235.58, £79.99

### **Rating Format**
- **Structure**: "X out of 5 stars"
- **Examples**: "4.9 out of 5 stars", "4.5 out of 5 stars"

## Data Statistics

*Based on preprocessing code analysis:*

### **Unique Value Counts**
```
uniq_id: 10,000
product_name: 9,964
manufacturer: 2,651
price: 2,625
number_available_in_stock: 89
number_of_reviews: 194
number_of_answered_questions: 19
average_review_rating: 19
amazon_category_and_sub_category: 255
customers_who_bought_this_item_also_bought: 8,755
description: 8,514
product_information: 9,939
product_description: 8,514
items_customers_buy_after_viewing_this_item: 6,749
customer_questions_and_answers: 910
customer_reviews: 9,901
sellers: 6,581
```

### **Missing Value Counts (Before Processing)**
```
manufacturer: 7
price: 1,435
number_available_in_stock: 2,500
number_of_reviews: 18
number_of_answered_questions: 765
average_review_rating: 18
amazon_category_and_sub_category: 690
customers_who_bought_this_item_also_bought: 1,062
description: 651
product_information: 58
product_description: 651
items_customers_buy_after_viewing_this_item: 3,065
customer_questions_and_answers: 9,086
customer_reviews: 21
sellers: 3,082
```

## Data Preprocessing Pipeline

### **1. Column Removal**
Dropped ineffective columns:
```
data.drop('customer_questions_and_answers', axis=1, inplace=True) # 9,086 missing (90.86%)
data.drop('number_of_answered_questions', axis=1, inplace=True) # No effect on recommendations
```


### **2. Probability-Based Imputation**
Applied to all categorical columns with missing values:

For each nominal/categorical column:
```
non_missing = data[column].dropna()
distribution = non_missing.value_counts(normalize=True)
imputed_values = np.random.choice(
distribution.index,
size=num_missing,
p=distribution.values
)
data.loc[data[column].isnull(), column] = imputed_values
```


**Rationale**: Maintains original data distribution better than mode imputation, especially important for recommendation systems where category frequencies influence results.

### **3. Label Encoding**
Applied to specific columns for ML compatibility:
```
le = LabelEncoder()
data['average_review_rating'] = le.fit_transform(data['average_review_rating'])
data['uniq_id'] = le.fit_transform(data['uniq_id'])
data['manufacturer'] = le.fit_transform(data['manufacturer'])
data['price'] = le.fit_transform(data['price'])
data['amazon_category_and_sub_category'] = le.fit_transform(data['amazon_category_and_sub_category'])
```

### **4. Feature Engineering**
Created new identifier:
```
label_encoder = LabelEncoder()
encoded_column = label_encoder.fit_transform(data['product_name'])
```

itemId added based on product_name encoding

Column renaming:
```
uniq_id → userId (for clarity in collaborative filtering)
```

## Final Dataset Structure

**After preprocessing:**
- **Total columns**: 15 (dropped 2 ineffective columns)
- **Missing values**: 0 (all imputed using probability-based method)
- **Target variable**: `average_review_rating` (label encoded)
- **User identifier**: `userId` (from `uniq_id`)
- **Item identifier**: `itemId` (from `product_name`)

## Access Instructions

To replicate this research:

1. **Download the original dataset**:
   ```
   git clone https://github.com/aksharpandia/miniamazon_data.git
   cp miniamazon_data/amazon_co-ecommerce_sample.csv ./data/
   ```


2. **Run the preprocessing pipeline**:
   ```
   python src/data_preprocessing.py
   ```



3. **Expected output**:
- **Cleaned dataset** with 0 missing values (probability-based imputation applied)
- **Label encoded categorical features** (manufacturer, price, categories, ratings)  
- **Enhanced dataset** with:
  - `userId` (from renamed `uniq_id`)
  - `itemId` (from encoded `product_name`)
  - All other processed columns ready for model training


## Data Usage Notes

- **Academic use**: Research conducted under academic fair use principles
- **Attribution**: Full credit to original data source repository
- **Privacy**: All identifiers are anonymous
- **Licensing**: Original repository provides no explicit license

## Research Implementation

**Preprocessing code location**: `src/data_preprocessing.py`  
**Supporting utilities**: `src/utils.py`  
**Research paper**: *"A Hybrid Recommender System for Amazon E-commerce Combining ALS Collaborative Filtering and Two-Tower Content-Based Filtering"*

---

**Note**: All statistics and processing details are based on the actual preprocessing pipeline applied to the original 10,000-product Amazon dataset. Sample data verification confirms data structure and format accuracy.


