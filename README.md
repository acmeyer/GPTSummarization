# GPTSummarization and Q&A

Get a quick summary of a document or url and ask questions about it!

## Try the bot!

As a bit of an experiment, I created [a GPT assistant for this repository](https://chat.openai.com/g/g-qxASQRhiF-article-reader). You can ask it questions about how this repository works and even ways to improve it. Pretty cool!

## How to use

Install the dependencies:

```
pip install -r requirements.txt
```

To generate a summary of a file, all you have to do is run the following command:

```
python main.py -f 'path/to/file.pdf'
```

To generate a summary of a url, run the following command:

```
python main.py -u 'https://www.example.com'
```

These commands will generate a summary of the file or url and both print it to the console as well as save it to the "data/summaries" directory. In addition, it will also prompt you to ask follow up questions if you have any about the document or url.

### Additional options

In addition to the file and url options, you can also specify the model to use and/or summary type. Here's how you could use these when summarizing a file:

```
python main.py -f 'path/to/file.pdf' -m 'gpt-3.5-turbo' -s 'short'
```
