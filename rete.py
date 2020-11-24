# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size= (7,7), strides = (1,1), padding = 'same',input_shape=(num_rows, num_columns, num_channels)))
model.add(LeakyReLU(alpha=leaky_relu_alpha))

#model.add(Conv2D(filters=32, kernel_size=(1,1), padding = 'same'))
#model.add(LeakyReLU(alpha=leaky_relu_alpha))
#model.add(MaxPooling2D(pool_size=2))
#model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3,3), strides = (1,1), padding = 'same'))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))


#model.add(Conv2D(filters=64, kernel_size=(1,1), padding = 'same'))
#model.add(LeakyReLU(alpha=leaky_relu_alpha))
#model.add(MaxPooling2D(pool_size=2))
#model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), strides = (1,1), padding = 'same'))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#model.add(Conv2D(filters=128, kernel_size=(1,1), padding = 'same'))
#model.add(LeakyReLU(alpha=leaky_relu_alpha))
#model.add(MaxPooling2D(pool_size=2))
#model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3,3),strides = (1,1), padding = 'same'))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(num_labels, activation='softmax'))


# Display model architecture summary 
model.summary()

#sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
opt = keras.optimizers.Adam(learning_rate=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=opt)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)



print(model)

