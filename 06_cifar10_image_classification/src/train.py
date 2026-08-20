import tensorflow as tf
from tensorflow.keras import layers,models
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.cifar10.load_data()
X_train,X_test=X_train/255.0,X_test/255.0
n=5000; X_val,y_val=X_train[-n:],y_train[-n:]; X_tr,y_tr=X_train[:-n],y_train[:-n]
aug=tf.keras.Sequential([layers.RandomFlip('horizontal'),layers.RandomRotation(.08),layers.RandomZoom(.1)])
model=models.Sequential([layers.Input((32,32,3)),aug,layers.Conv2D(32,3,activation='relu'),layers.BatchNormalization(),layers.Conv2D(32,3,activation='relu'),layers.MaxPooling2D(),layers.Dropout(.25),layers.Conv2D(64,3,activation='relu'),layers.BatchNormalization(),layers.Conv2D(64,3,activation='relu'),layers.MaxPooling2D(),layers.Dropout(.3),layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dropout(.4),layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
cb=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint('best_model.keras',save_best_only=True)]
model.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=30,batch_size=128,callbacks=cb)
print('test=',model.evaluate(X_test,y_test,verbose=0))