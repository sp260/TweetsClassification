from TwitterAPI import TwitterAPI
api = TwitterAPI('APIKey','APISecretKey','AccessToken','AccessTokenSecret')

r = api.request('search/tweets', {'q':'Macron', 'count':5000, 'tweet_mode': 'extended'})

corpus = ''

for item in r.get_iterator():
	tweetId = item['id']
	if 'retweeted_status' in item:
		tweetText = item['retweeted_status']['full_text']
		corpus += '(' + str(tweetId) + ', ???)' + tweetText.replace('\n', ' ') + '\n'
	else:
		tweetText = item['full_text']
		corpus += '(' + str(tweetId) + ', ???)' + tweetText.replace('\n', ' ') + '\n'

file = open('TweetsAboutMacron.txt','w')
file.write(corpus) 
file.close()