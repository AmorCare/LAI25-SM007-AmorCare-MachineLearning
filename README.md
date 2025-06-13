# AmorCare Machine Learning Part

## Dataset
Data yang digunakan dibagi menjadi tiga bagian, yaitu data train, data validasi, dan data test. Jumlah seluruh data yang telah dikumpulkan sebesar 3354 data. [Link Dataset](https://drive.google.com/drive/folders/1kfWBLaNuRVLJnB7ZHpAhvwKIFTytRAXJ?usp=drive_link)

|   Jenis Kulit   | Jumlah | 
|:---------------:|:------:|
|  Berjerawat  |  1118  | 
|  Berminyak  |  1118  | 
|  Normal  |  1118  | 

## Model
Model yang dibangun menggunakan algoritma CNN dengan menerapkan metode transfer learning dan digunakan pula MobileNetV2 sebagai model dasar. Untuk meningkatkan akurasi model, dilakukan proses fine-tuning serta penambahan lapisan agar model dapat beradaptasi secara khusus dengan dataset jenis kulit yang dimiliki. Pendekatan ini membantu model mengenali jenis kulit secara lebih tepat dan efektif, sesuai dengan kebutuhan AmorCare.

     ```
     base_model = tf.keras.applications.MobileNetV2(include_top=False,
                        weights='imagenet',
                        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = True

    # Fine-tune
    fine_tune_at = len(base_model.layers) - 40 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    def create_model(base_model):
        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        x = base_model(inputs, training=True)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)
     ```
## Model Performance

![image](https://github.com/user-attachments/assets/0bfe86c1-03a9-4ae4-b516-5d2898012cda)
Akurasi training dan validasi berhasil meningkat seiring bertambahnya epoch, dengan loss yang menurun secara signifikan. Hal ini menunjukkan bahwa model MobileNetV2 berhasil belajar dengan baik.

![image](https://github.com/user-attachments/assets/e91fb63c-c6bc-4714-91ef-c1582f2c8df4)

![image](https://github.com/user-attachments/assets/a5dbd932-c4eb-495b-80b3-30d524652799)

     Classification Report:
                   precision    recall  f1-score   support
     
        berminyak       0.95      0.88      0.91       112
          jerawat       0.92      0.93      0.92       112
           normal       0.88      0.95      0.91       112
     
         accuracy                           0.92       336
        macro avg       0.92      0.92      0.92       336
     weighted avg       0.92      0.92      0.92       336

Model berhasil mencapai akurasi keseluruhan sebesar 92%, dengan nilai precision, recall, dan f1-score yang seimbang pada ketiga kelas. Hal ini menunjukkan bahwa performa klasifikasi yang stabil dan andal.

## Requirements
- tensorflow
- numpy
- matplotlib
- seaborn
- ipywidgets
- scikit-learn

## Tools
- Visual Studio Code
- Google Drive
- Python Libraries
  - TensorFlow: Digunakan untuk melatih model machine learning.
  - NumPy: Digunakan untuk memanipulasi data.
  - Matplotlib and Seaborn: Digunakan untuk memvisualisasi data
  - Scikit-learn : Digunakan untuk mengevaluasi model
  - ipywidgets and IPython: Digunakan untuk membuat widget interaktif dan meningkatkan pengalaman interaktif dalam lingkungan Python.



