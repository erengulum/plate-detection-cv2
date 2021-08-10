import cv2
import numpy as np
import math
from skimage.feature import peak_local_max



def getEdgedimage(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75,75)  # it is recommended to use  d=9 for offline applications that need heavy noise filtering. For simplicity, you can set the 2 sigma values to be the same
    edges = cv2.Canny(filtered, 120, 250)  #Second and third arguments are  minVal and maxVal respectively

    return edges



def listToDictionary(inputlist): # Convert the given 2d array to a dictionary
    d = {}
    for elem in inputlist:
        if elem[1] in d:
            d[elem[1]].append(elem[0])
        else:
            d[elem[1]] = [elem[0]]
    return d



def minmaxPerpendicular(perpDict):
    '''Plaka harflerindeki dikey kısımlardan etkilenmemesi için dik kenarların en sağındaki ve en solundakini aldık (arada kalanlar genelde harfler oluyor)'''
    minMaxPerpDict = {}
    for key, value in perpDict.items():
        minMaxPerpDict[key] = [min(value)]
        minMaxPerpDict[key].append(max(value))

    return minMaxPerpDict



def deleteNotPerpendicularLines(paralelDict,perpendicularDict):
    '''Delete angles that have less than two perpendicular edges'''
    deletedPerpendicular = []
    for key, value in perpendicularDict.items():
        if len([item for item in value if item]) < 2:  # theta'nın sayısı 2'den azsa paralellik yoktur bu yüzden bunu sileceğiz
            deletedPerpendicular.append(key)

    for i in deletedPerpendicular:
        perpendicularDict.pop(i)

    paralelLinesToBeDeleted = []
    for key in paralelDict.keys():
        if ((key-90) not in perpendicularDict) and ((key+90) not in perpendicularDict):
            paralelLinesToBeDeleted.append(key)

    for i in paralelLinesToBeDeleted:
        paralelDict.pop(i)


    return paralelDict, perpendicularDict




def dropNonParalelLines(lineDict):
    '''Edges of car plates are mostly paralells. So, i deleted the lines that doesn't have any other parallel line'''
    willdeleted = []
    for key, value in lineDict.items():
        if len([item for item in value if item]) <2:  # theta'nın sayısı 2'den azsa paralellik yoktur bu yüzden bunu sileceğiz
            willdeleted.append(key)

    for i in willdeleted:
        lineDict.pop(i)

    return lineDict




def findPerpendicularLines(houghMatrix, parallelLinesDic,threshold=30):
    '''License plates consist of two parallel sides and two sides perpendicular to them. We will find the edges perpendicular to the parallel sides taken as input here. '''
    perp_lines = {}
    for keys in parallelLinesDic.keys():
        for i in range(0, houghMatrix.shape[0], 1):
            positive_perp = int(keys + 90)
            negative_perp = int((keys - 90))

            if (positive_perp <= 180) and (houghMatrix[i][positive_perp] > threshold):  # brightness değeri
                if positive_perp in perp_lines:
                    perp_lines[positive_perp].append(i)
                else:
                    perp_lines[positive_perp] = [i]
            if (negative_perp >= 0) and ( houghMatrix[i][negative_perp] > threshold):  # brightness değeri
                if negative_perp in perp_lines:
                    perp_lines[negative_perp].append(i)
                else:
                    perp_lines[negative_perp] = [i]

    return perp_lines



def createRhoThetaSpace(edged_image):
    '''This function creates a Hough Accumulator/Rho-Theta space'''
    X, Y = np.shape(edged_image)
    theta_range = 181  # 181 columns
    diagonalLength = int((X ** 2 + Y ** 2) ** (0.5))  # hypotenus
    rhoThetaSpace = np.zeros((diagonalLength, theta_range))  # dtype=np.float32
    return rhoThetaSpace



def houghVote(edged_image, HoughMatrix): #Voting operation for the hough accumulator
    X, Y = np.shape(edged_image)
    for x in range(X):
        for y in range(Y):
            if edged_image[x][y] > 0:
                for theta in range(0, 181, 1):
                    rho = int(x * np.sin(np.deg2rad(theta)) + y * np.cos(np.deg2rad(theta)))
                    HoughMatrix[rho][theta] += 1

    return HoughMatrix




def candidateLines(houghMatrix):
    '''After finding the local maximum points in Hough Matrix with the help of the built-in function, we will convert it to the dictionary structure.'''
    peaks = peak_local_max(houghMatrix, min_distance=10, threshold_rel=0.5)
    lineDict = listToDictionary(peaks)  # list to dictionary

    '''Delete the non-paralel lines  and find the perpendicular lines'''
    paralelLinesDic = dropNonParalelLines(lineDict)
    perpendicular_lines = findPerpendicularLines(houghMatrix, paralelLinesDic)

    paralelLinesDic, perpendicular_lines = deleteNotPerpendicularLines(paralelLinesDic, perpendicular_lines)
    perpendicular_lines = minmaxPerpendicular(perpendicular_lines)

    return paralelLinesDic, perpendicular_lines



def detectPlate(paralel_lines, perp_lines, img): #Conducting plate detection by using auxilliary plotLine function which is defined below
    length = np.shape(img)[1]
    plotLine(img, paralel_lines, length)
    plotLine(img, perp_lines, length)
    cv2.imshow("Plate Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plotLine(img, line_dictionary, length):
    for key, value in line_dictionary.items(): #key=theta and value is the list of rhos
        for rho in value:
            a = math.cos(np.deg2rad(key))
            b = math.sin(np.deg2rad(key))
            x = a * rho
            y = b * rho
            pt1 = (int(x + length * (-b)), int(y + length * (a)))
            pt2 = (int(x - length * (-b)), int(y - length * (a)))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2, lineType=cv2.LINE_AA)



'''STARTING POINT'''
print("Please enter the full path of the image:" )
imagepath= str(input())

image = cv2.imread(imagepath)
edgedImage = getEdgedimage(image)

#rhoThetaArray for the Hough Space
rhoThetaArray = createRhoThetaSpace(edgedImage)
filledHoughSpace = houghVote(edgedImage, rhoThetaArray)

#plate_recognition=
paralelDict,perpendicularDict = candidateLines(filledHoughSpace)
detectPlate(paralelDict, perpendicularDict, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
