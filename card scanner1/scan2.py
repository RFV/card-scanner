# import the necessary packages
from imageutils.transform import four_point_transform
from imageutils import imutils
from skimage.filters import threshold_adaptive
import argparse
import cv2
import numpy as np

scale_factor = 500
cv2version = int(cv2.__version__.split(".")[0])

class EdgeNotFound(Exception): pass

def _findContours(p1, p2, p3):
	if cv2version is 3: return cv2.findContours(p1, p2, p3)[1]
	elif cv2version is 2: return cv2.findContours(p1, p2, p3)[0]

def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required = True, help = "Path to the image to be scanned")
	ap.add_argument("-o", "--output", required = True, help = "Path to the image to be saved, or 'screen' for showing each step on screen")
	
	ap.add_argument("-f", "--filter", required = False, help = "Return filtered image. Requires canny1,canny2. 10,10")
	ap.add_argument("-c", "--contrast", required = False, help = "Return contrasted image. Requires thresh1,thresh2. 180,200")
	ap.add_argument("-m", "--merge", required = False, help = "Return contrast merged image. Requires distance. 50")
	
	ap.add_argument("-p", "--overlay", required = False, help = "Return border recognized image. Requires polydp. 0.03")
	ap.add_argument("-P", "--cropped", required = False, help = "Returns the payload image. Requires polydp. 0.03" )

	args = vars(ap.parse_args())

	work_image = cv2.imread(args["input"])
	original_image = work_image.copy()
	output_image_name = args["output"]

	if output_image_name == 'screen': show = True
	else: show = False

	if args["filter"]:
		canny1, canny2 = args['filter'].split(',')
		canny1 = int(canny1)
		canny2 = int(canny2)
		work_image = filter_image(work_image, canny1=canny1, canny2=canny2, show=show)

	elif args["contrast"]:
		thresh1, thresh2 = args['contrast'].split(',')
		thresh1 = int(thresh1)
		thresh2 = int(thresh2)
		work_image = contrast_image(work_image, thresh1=thresh1, thresh2=thresh2, show=show)

		if args["merge"]:
			distance = int(args['merge'])
			work_image = merge_image_contours(work_image , distance=distance, show=show)

	if args["overlay"]:
		polydb = float(args["overlay"])
		try:
			work_image = overlay(original_image, get_contours(work_image, polydb=polydb), show=show)
		except EdgeNotFound: print 'edge not found'

	elif args["cropped"]:
		polydb = float(args["cropped"])
		try:
			work_image = cropped_image(original_image, get_contours(work_image, polydb=polydb), show=show)
		except EdgeNotFound: print 'edge not found'

	if not show: cv2.imwrite(output_image_name, work_image)

def filter_image(image, canny1=10, canny2=10, show=False):
	# compute the ratio of the old height to the new height, and resize it
	image = imutils.resize(image, height=scale_factor, interpolation=cv2.INTER_CUBIC)

	# convert the image to grayscale, blur it, and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, canny1, canny2)

	# show the image(s)
	if show:
		cv2.imshow("Edged", edged)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return edged

def contrast_image(image, thresh1=180, thresh2=200, show=False):
	image = imutils.resize(image, height=scale_factor)
	# convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.GaussianBlur(gray, (5, 5), 0)

	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(gray2, thresh1, thresh2, cv2.THRESH_BINARY)[1]
	thresh2 = cv2.erode(thresh, None, iterations=2)
	thresh3 = cv2.dilate(thresh2, None, iterations=2)

	if show is True: #this is for debugging puposes
		cv2.imshow("Contrast", thresh3)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return thresh

def merge_image_contours(image, distance=50, show=False):
	'''before calling this method make sure the image has allready been contrasted -> otherwise the method may run for a very long time'''
	def find_if_close(cnt1, cnt2):
		row1, row2 = cnt1.shape[0], cnt2.shape[0]
		for i in xrange(row1):
			for j in xrange(row2):
				dist = np.linalg.norm(cnt1[i]-cnt2[j])
				if abs(dist) < distance:
					return True
				elif i==row1-1 and j==row2-1:
					return False

	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# ret, thresh = cv2.threshold(image,127,255,0)
	thresh = image.copy()
	# if cv2version == 3: im2, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
	contours = _findContours(thresh, cv2.RETR_EXTERNAL, 2)
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))

	for i, cnt1 in enumerate(contours):
		x = i    
		if i != LENGTH-1:
			for j, cnt2 in enumerate(contours[i+1:]):
				x = x+1
				dist = find_if_close(cnt1, cnt2)
				if dist == True:
					val = min(status[i], status[x])
					status[x] = status[i] = val
				else:
					if status[x]==status[i]:
						status[x] = i+1

	unified = []
	maximum = int(status.max())+1
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)

	# cv2.drawContours(image, unified,-1,(0,255,0),2)
	cv2.drawContours(thresh, unified,-1,255,-1)

	if show is True: #this is for debugging puposes
		cv2.imshow("123", thresh)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return thresh

def get_contours(image, polydb=0.03, contour_range=5, show=False):
	# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
	# if cv2version == 3: im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = _findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:contour_range]

	# loop over the contours
	screenCnt = None
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True) #finds the Contour Perimeter
		approx = cv2.approxPolyDP(c, polydb * peri, True)

		# if our approximated contour has four points, then we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	if screenCnt is None: raise EdgeNotFound()

	# sometimes the algorythm finds a strange non-convex shape. The shape conforms to the card but its not complete, so then just complete the shape into a convex form
	if not cv2.isContourConvex(screenCnt):
		screenCnt = cv2.convexHull(screenCnt)
		x,y,w,h = cv2.boundingRect(screenCnt)
		screenCnt = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])

	if show: #this is for debugging puposes
		new_image = image.copy()
		cv2.drawContours(new_image, [screenCnt], -1, (255, 255, 0), 2)
		cv2.imshow("Contour1 image", new_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return screenCnt

def overlay(image, contours, show=False):
	new_image = imutils.resize(image, height=scale_factor)
	cv2.drawContours(new_image, [contours], -1, (2, 2, 251), 2)
	if show: #this is for debugging puposes
		cv2.imshow("Contour1 image", new_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return new_image

def cropped_image(image, contours, bw=True, show=False):
	ratio = image.shape[0] / float(scale_factor)
	warped = four_point_transform(image, contours.reshape(4, 2) * ratio)
	if bw:
		# convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		warped = threshold_adaptive(warped, 251, offset = 10)
		warped = warped.astype("uint8") * 255

	if show: #this is for debugging puposes
		cv2.imshow("Payload", warped)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return warped

def i1(canny1=5, canny2=5, polydb=0.04, contour_range=5, image_file='images\\WP_20160531_11_32_28_Rich_LI.jpg'):
	image = cv2.imread(image_file)
	filtered_image = filter_image(image, canny1=canny1, canny2=canny2)
	contours = get_contours(filtered_image, polydb=polydb, contour_range=contour_range)
	overlay(image, contours, show=True)
	cropped_image(image, contours, show=True)

def i2(canny1=10, canny2=10, polydb=0.03, image_file='images\\WP_20160531_11_33_24_Rich_LI.jpg'):
	image = cv2.imread(image_file)
	filtered_image = filter_image(image, canny1=canny1, canny2=canny2)
	contours = get_contours(filtered_image, polydb=polydb)
	overlay(image, contours, show=True)
	cropped_image(image, contours, show=True)

def i3(thresh1=180, thresh2=200, distance=50, polydb=0.03, image_file='images\\WP_20160531_11_36_09_Rich_LI.jpg'):
	image = cv2.imread(image_file)
	contrasted_image = contrast_image(image, thresh1=thresh1, thresh2=thresh2)
	gelled_image = merge_image_contours(contrasted_image, distance=distance)
	contours = get_contours(gelled_image, polydb=polydb)
	overlay(image, contours, show=True)
	cropped_image(image, contours, show=True)

def i4(canny1=10, canny2=10, polydb=0.03, image_file='images\\WP_20160531_11_38_57_Rich_LI.jpg'):
	image = cv2.imread(image_file)
	filtered_image = filter_image(image, canny1=canny1, canny2=canny2)
	contours = get_contours(filtered_image, polydb=polydb)
	overlay(image, contours, show=True)
	cropped_image(image, contours, show=True)

if __name__ == '__main__':
	main()