import os
from os import walk
from os.path import isfile
import requests
import sys
import cv2
import numpy as np
from PIL import Image


def dilation(img):
	se = [[-1, -1], [-1, 0], [-1, 1], [0, 0], [0, 1], [0, -1], [1, 0], [1, -1], [1, 1]]
	m = img.shape[0]
	n = img.shape[1]
	newimg = np.array([np.array([np.uint8(255) for j in range(n+30)]) for i in range(m+30)])
	for i in range(m):
		for j in range(n):
			if(img[i][j] == 255):
				continue
			for k in se:
				newimg[i+k[0]][j+k[1]] = np.uint8(0)
	return newimg


class DSU:
	def __init__(self, n):
		self.par = [i for i in range(n)]
		self.Rank = [0]*n

	def Find(self, i):
		if(self.par[i] == i):
			return i 
		f = self.Find(self.par[i])
		self.par[i] = f 
		return f

	def Union(self, u, v):
		x = self.Find(u)
		y = self.Find(v)
		if(x == y):
			return
		if(self.Rank[x] <= self.Rank[y]):
			self.par[x] = y 
			if(self.Rank[x] == self.Rank[y]):
				self.Rank[y] += 1 
		else:
			self.par[y] = x

class Boundary:
	def __init__(self, i, j, k, l):
		self.tlx = i 
		self.tly = j
		self.brx = k 
		self.bry = l 

	def width(self):
		return self.bry - self.tly + 1

	def height(self):
		return self.brx - self.tlx + 1

	def maxboundary(self, x, y):
		self.tlx = min(self.tlx, x);
		self.tly = min(self.tly, y);
		self.brx = max(self.brx, x);
		self.bry = max(self.bry, y);

def generate_images(img):
	img = dilation(img)
	img = dilation(img)

	m = img.shape[0]
	n = img.shape[1]
	tot = m * n 
	label = [[-1 for j in range(n)] for i in range(m)]
	boundary = [-1]*tot
	d = DSU(tot)
	curr_label = 0

	# pass 1
	for i in range(m):
		for j in range(n):
			if(img[i][j] == 255):
				continue
			rr, rc, tr, tc = i-1, j, i, j-1

			if(rr >= 0 and tc >= 0 and img[rr][rc] == 0 and img[tr][tc] == 0):
				label[i][j] = label[rr][rc]
				d.Union(label[rr][rc], label[tr][tc])

			elif(rr >= 0 and img[rr][rc] == 0):
				label[i][j] = label[rr][rc]

			elif(tc >= 0 and img[tr][tc] == 0):
				label[i][j] = label[tr][tc]

			else:
				label[i][j] = curr_label
				curr_label += 1

	# pass 2
	for i in range(m):
		for j in range(n):
			if(label[i][j] == -1):
				continue 
			label[i][j] = d.Find(label[i][j])

	s = set()	# stores unique labels

	# compute boundaries
	for i in range(m):
		for j in range(n):
			if(label[i][j] == -1):
				continue
			s.add(label[i][j])
			if(boundary[label[i][j]] == -1):
				boundary[label[i][j]] = Boundary(i, j, i, j)

			else:
				boundary[label[i][j]].maxboundary(i, j)

	# remove small components
	small = []
	for l in s:
		if(boundary[l].height() < 10):
			small.append(l)

	for x in small:
		s.remove(x)



	# let's try to separate a component

	s = list(s)
	s.sort(key = lambda a: (boundary[a].tly))

	def getblue(img):
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if(img[i][j][0] == img[i][j][1] == img[i][j][2] == 0):
					img[i][j][0] = 255
		return img


	global names

	mh, mw = 0, 0 
	for x in s:
		mw = max(mw, boundary[x].width())
		mh = max(mh, boundary[x].height())
	ni = 0
	for x in s:
		cur_l = x
		aimg = np.array([np.array([np.uint8(255) for j in range(mw)]) for i in range(mh)])
		offset = mh - boundary[cur_l].height()
		for i in range(boundary[cur_l].height()):
			for j in range(boundary[cur_l].width()):
				if(img[boundary[cur_l].tlx + i][boundary[cur_l].tly + j] == 0):
					aimg[offset + i][j] = np.uint8(0)
		aimg = getblue(aimg)
		cv2.imwrite('./pics/' + str(names[ni]) + '.png', aimg)
		ni += 1


def convert_text():
	# text = 'Hey this is Sayantan Chakraborty. I developed this cool system.\nHope you find it useful.'
	# letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
	f = open('input.txt', 'r')
	text = f.read()
	global names
	letters = names
	images = [Image.open('pics/{}.png'.format(i)) for i in letters]

	widths, heights = zip(*(i.size for i in images))

	newline = sum([1 if x == '\n' else 0 for x in text])
	#total_width = sum(widths)
	# total_width = int(max(widths)*(len(text)/newline+1))
	total_width = 33 * max(widths)
	max_height = max(heights) 
	total_height = max_height * 35

	new_im = Image.new('RGB', (total_width, total_height), 'white')

	x_offset = 50
	y_offset = max_height + 20
	count = 0
	nc = 0
	pc = 0

	for i in range(len(text)):
		im = images[0]
		if(text[i] != ' ' and text[i] != '\n'):
			im = images[letters.index(text[i])]
			new_im.paste(im, (x_offset,y_offset))

		if(x_offset == 50 and text[i] == ' '):
			continue

		x_offset += im.size[0]-15
		count  += 1 
		if(text[i] == '\n' or count == 34):
			y_offset += max_height + 20
			x_offset = 50
			count = 0
			nc += 1
			if(nc == 25):
				new_im.save(str(pc) + '.png')
				x_offset = 50
				count = nc = 0
				y_offset = max_height + 20
				pc += 1
				new_im = Image.new('RGB', (total_width, total_height), 'white')

	
	new_im.save(str(pc) + '.png')
	os.system('convert *png output/mypdf.pdf')
	os.system('rm *png')

if __name__ == '__main__':
	img = cv2.imread('sample.jpeg', 0)
	img = cv2.resize(img, (800, 800))
	

	for i in range(len(img)):
		for j in range(len(img[0])):
			if(img[i][j]<95):
				img[i][j] = 0 
			else:
				img[i][j] = 255

	#cv2.imshow('img', img)
	#cv2.waitKey(0)

	names = ['8', 'Y', 'O', 'u', 'E', 'a', 'k', '9', 'z', 'v', 'F', 'P', 'l', 'b', '0', '.', 'Q', 'G', 'w', 'm', 'C',
	'1', 'R', 'H', 'x', 'n', 'd', 'S', '2', 'I', 'y', 'o', 'e', 'T', '3', 'J', 'Z', 'p', 'f', '4', 'U','A', 'K', 'q', 'g', '5',
	'V', 'r', 'L', 'B', 'h', '6', 'W', 'M', 's', 'c', 'i', '7', 'X', 'N', 't', 'D', 'j']

	# generate_images(img) 	# uncomment this line to generate images of your handwriting
	convert_text()
	print('Done... Output generated as: output/mypdf.pdf')