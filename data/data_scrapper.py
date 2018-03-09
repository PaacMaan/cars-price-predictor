from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv


def get_ads_urls():
	urls_list = []
	# define the basic url to crawl on
	basic_url = "https://www.avito.ma/fr/maroc/voitures-à_vendre?sp=1&o="
	# loop over the paginated urls
	for i in range(1,100):
		# get the page url
		url = basic_url+str(i)
		# get the request response
		r  = requests.get(url)
		data = r.text
		# transform it to bs object
		soup = BeautifulSoup(data, "lxml")
		# loop over page links
		for div in soup.findAll('div', {'class': 'item-img'}):
		    a = div.findAll('a')[0]
		    urls_list.append(a.get('href'))
		

	df = pd.DataFrame(data={"url": urls_list})
	df.to_csv("./ads_urls.csv", sep=',',index=False)
    		
def scrap_ad_data(ad_url):
	r = requests.get(ad_url)
	data = r.text
	soup = BeautifulSoup(data, "html.parser")
	target_component = soup.findAll("h2",  {"class": ["font-normal", "fs12", "no-margin", "ln22"]})
	# create a list that will hold our component data
	results = []
	for i in target_component:
		results.append(''.join(i.findAll(text=True)).replace('\n',''))
	return results
	
def write_data_to_csv(data):
	with open("output.csv", "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(data)

if __name__ == '__main__':
	# get the ads urls and save them in a file
	# get_ads_urls()
	# read the saved urls file as a dataframe 
	urls_data = pd.read_csv("ads_urls.csv")
	# create  a list that will hold the final data
	final_result = []
	i = 0
	# loop over the dataframe
	for index, row in urls_data.iterrows():
		final_result.append(scrap_ad_data(row['url']))
		i += 1
		if i%34 == 0:
			print("page ",int(i/34), "done")

	# now that we have all the data we can write it in a csv file
	write_data_to_csv(final_result)

	