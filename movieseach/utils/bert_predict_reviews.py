import ktrain
from ktrain import text
import glob


(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder('aclImdb',
                                                                       maxlen=500,
                                                                       preprocess_mode='bert',
                                                                       train_test_names=['train',
                                                                                         'test'],
                                                                       classes=['pos', 'neg'])
model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)
learner.fit_onecycle(2e-5, 2)  # train for 2 epochs
predictor = ktrain.get_predictor(model, preproc)
predictor.save('/models/predictor')

predictor = ktrain.load_predictor('/models/predictor')
dataset = 'aclImdb/train/unsup'
file_list = glob.glob(dataset + "/*.txt")
results = open("train_labels.txt", "w")
for file in file_list:
    review_text = open(file, "r", encoding="utf-8").readlines()[0]
    predict = predictor.predict(review_text)
    results.write(predict+'\n')
results.close()
