# textgenrnn-chinese
Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code.
代码源于https://github.com/minimaxir/textgenrnn
源代码支持英文语料，我进行了修改支持中文语料

运行test.py文件，文件内容如下：

from textgenrnn import textgenrnn
###初始化文本生成实例
# textgen = textgenrnn(weights_path='textgenrnnword_weights.hdf5',vocab_path='textgenrnnword_vocab.json',name='textgenrnnword')
textgen = textgenrnn(weights_path='textgenrnnchar_weights.hdf5',vocab_path='textgenrnnchar_vocab.json',name='textgenrnnchar')
##预训练模型基础上训练新文本
# textgen.train_from_file('datasets/test.txt', num_epochs=5,word_level=False)
# textgen.generate(5,temperature=[0.5])
# ###训练新模型
# textgen.reset()
# textgen.train_from_file('datasets/hacker_news_10.txt', num_epochs=10,gen_epochs=5,train_size=0.8,dropout=0.2,new_model=True,word_level=False)
# textgen.generate(5,temperature=[0.5])
# ##指定开头生成句子
# textgen.generate_samples(prefix='借钱',temperatures=[0.5])
# textgen.generate(prefix='宣',temperature=[0.5])
#
# ##相似计算
# print(textgen.similarity(text='套路贷业务员会怎么判',texts=['套路贷业务员会怎么判','怎么判断和防范“套路贷','业务员如何解决被拖欠款项'],use_pca=False))
###生成向量
texts = ['套路贷业务员会怎么判',
            '怎么判断和防范“套路贷']
word_vector = textgen.encode_text_vectors(texts, pca_dims=None)
print(word_vector)
print(word_vector.shape)
