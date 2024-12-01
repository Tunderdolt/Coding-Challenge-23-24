#Computation Challenge - Sam Binns 11/1/24 thru 27/02/24
#A quick guide to terms used in this code
#i values will always refer to the column in the array or the 'x-value'
#j values will always refer to the row in the array or the 'y-value'
#bmp array refers to the array generated from the given bmp image by the convert_bmp_to_array() function line 17
#array coordinates is the given index position of a value in an array
#contiguous points refers to points within a 3x3 square of an array coordinate with the given array coordinate in the centre of that square

#For image 1 midpoint is [524, 287]
#For image 2 midpoint is [541, 276]
 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_rhombus_midpoint = np.array([541, 276])

def convert_bmp_to_array( filename: str, width : int = 1000, height : int = 600 ):
    """Take the file filename.bmp, of pre-specified height and width (in pixels) and transform it into 
    the equivalent numpy array.
    INPUTS:
    filename: a string containing the name of the file
    width: width of the image, in pixel (keep the default values)
    height: height of the image, in pixel (keep the default values)
    OUTPUT:
    data: a numpy array containing either 0 or 255, if the corresponding pixel is white or black, respectively
    """
    
    #First a check that the file has the correct extension and is of type bitmap
    if filename[-4:] != ".bmp":
        raise ValueError("The file must be in .bmp format")

    #Image is a specific Python object of the PIL library
    image = Image.open(filename)

    # Convert the image to black and white
    bw_image = image.convert('1')

    # Get the pixel values as a list of zeros and ones
    pixels = list(bw_image.getdata())

    # Convert the pixel values to a list of integers
    zeros_and_ones = [int(p) for p in pixels]
    tot = len( zeros_and_ones )

    #Transform the list into the appropriate numpy array
    data = np.array( zeros_and_ones ).reshape( ( height, width ))
    
    data = -data + 255
    
    return data


class ImageProcessor():
    """
    A class which processes an array to remove surperfluous data.

    Attributes
    ----------
    data : ndarray
        The array generated from a bmp image
    jMax : int
        The number of elements along the 'x-axis' of the array
    iMax : int
        The number of elements along the 'y-axis' of the array
    
    Methods
    -------
    pooling()
        Returns an array generated from an original array with data outside the obvious shape reduced
    clean_image(N)
        Runs the function pooling N number of times and returns the resulting array
    """

    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        data : ndarray
            The array generated from a bmp image
        """

        #Sets values of bmp array between 0 and 1
        self.data = data / 255

        self.jMax = np.shape(self.data)[0] - 1
        self.iMax = np.shape(self.data)[1] - 1
    
    def pooling(self):
        """Returns an array generated from an original array with data outside the obvious shape reduced
        """

        pooledArray = np.zeros(np.shape(self.data))

        #A set of for loops that take the average of contiguos points to an index position coordinate and passes the averaged value into the empty array pooledArray
        
        # Corner cases
        #Corners are unique as they each only have 4 contiguous points
        pooledArray[0, 0] = (self.data[0, 0] + self.data[0, 1] + self.data[1, 0] + self.data[1, 1]) / 4
        pooledArray[self.jMax, 0] = (self.data[self.jMax, 0] + self.data[self.jMax, 1] + self.data[self.jMax-1, 0] + self.data[self.jMax-1, 1]) / 4
        pooledArray[0, self.iMax] = (self.data[0, self.iMax] + self.data[1, self.iMax] + self.data[0, self.iMax-1] + self.data[1, self.iMax-1]) / 4
        pooledArray[self.jMax, self.iMax] = (self.data[self.jMax, self.iMax] + self.data[self.jMax-1, self.iMax] + self.data[self.jMax, self.iMax-1] + self.data[self.jMax-1, self.iMax-1]) / 4

        # Edge cases
        #Edge cases are unique as they have only 6 contiguous points
        for i in range(1, self.iMax):
            pooledArray[0, i] = (self.data[0, i] + self.data[1, i] + self.data[0, i+1] + self.data[1, i+1] + self.data[0, i-1] + self.data[1, i-1]) / 6
            pooledArray[self.jMax, i] = (self.data[self.jMax, i] + self.data[self.jMax, i+1] + self.data[self.jMax, i-1] + self.data[self.jMax-1, i] + self.data[self.jMax-1, i+1] + self.data[self.jMax-1, i-1]) / 6

        for j in range(1, self.jMax):
            pooledArray[j, 0] = (self.data[j, 0] + self.data[j, 0] + self.data[j-1, 0] + self.data[j+1, 0] + self.data[j-1, 1] + self.data[j+1, 1]) / 6
            pooledArray[j, self.iMax] = (self.data[j, self.iMax] + self.data[j, self.iMax-1] + self.data[j-1, self.iMax] + self.data[j+1, self.iMax] + self.data[j-1, self.iMax-1] + self.data[j+1, self.iMax-1]) / 6

        # Centre cases
        #Within the edges of the array each index coordinate has 9 contiguous points
        for i in range(1, self.iMax):
            for j in range(1, self.jMax):
                pooledArray[j, i] = (self.data[j, i] + self.data[j+1, i] + self.data[j, i-1] + self.data[j, i+1] + self.data[j+1, i-1] + self.data[j+1, i+1] + self.data[j-1, i] + self.data[j-1, i-1] + self.data[j-1, i+1]) / 9

        return np.around(pooledArray, decimals=0)
    
    def clean_image(self, N: int):
        """Runs the function pooling N number of times and returns the resulting array
        
        Parameters
        ----------
        N : int
            a number which determines the number of times the pooling method runs
        """

        for i in range(0, N):
            cleanedImage = self.pooling()
        
        return cleanedImage

class Rhombus():
    """
    A class defining a rhombus as vertices and an ndarray

    Methods
    -------
    random_rhombus_vertices_generator()
        Returns vertices of a randomly generated rhombus
    rhombus_array_generator(rhombus_vertices)
        Returns an array generated from the vertices passed in
    """

    def random_vertices_generator():
        """Returns vertices of a randomly generated rhombus centered around the centre of the image
        """

        #Generates a random vector
        half_AB = np.array([np.random.randint(50, 300), np.random.randint(50, 175)])

        #Random vector is added and subtracted to midpoint of image to generate points A and B
        A = image_rhombus_midpoint - half_AB
        B = image_rhombus_midpoint + half_AB

        beta = np.random.rand()

        #Perpendicular bisector of AB taken and multiplied by random scalar beta and added to midpoint to produce points C and D
        C = image_rhombus_midpoint + np.array([(beta * half_AB[1]), (-beta * half_AB[0])])
        D = image_rhombus_midpoint - np.array([(beta * half_AB[1]), (-beta * half_AB[0])])

        rhombus_vertices = np.array([A, B, C, D])

        return rhombus_vertices

    def array_generator(rhombus_vertices: np.ndarray):
        """Returns an array generated from the vertices passed in

        Parameters
        ----------
        rhombus_vertices : ndarray
            An array containing the vertices describing a rhombus
        """

        #Orders the array of vertices from lowest to highest y value
        sorted_rhombus = rhombus_vertices[rhombus_vertices[:, 1].argsort()]
        sorted_rhombus = sorted_rhombus.round()
        sorted_rhombus = sorted_rhombus.astype('int')
        rhombus_array = np.zeros((600, 1000))

        #Set the lowest y-value coordinate to p_1
        p_1 = sorted_rhombus[0]

        #Set the highest y-value coordinate to p_4
        p_4 = sorted_rhombus[3]

        #Set the left-most middle value to p_2 and right-most to p_3
        p_2 = sorted_rhombus[sorted_rhombus[(1, 2), 0].argsort() + 1][0]
        p_3 = sorted_rhombus[sorted_rhombus[(1, 2), 0].argsort() + 1][1]

        #Calculate the lines between respective points
        l_12 = (p_2 - p_1) / abs(p_2[1] - p_1[1])
        l_13 = (p_3 - p_1) / abs(p_3[1] - p_1[1])
        l_24 = (p_4 - p_2) / abs(p_4[1] - p_2[1])
        l_34 = (p_4 - p_3) / abs(p_4[1] - p_3[1])
        
        #A check to see if lines are horizontal
        l_12_1_isnan = np.isnan(l_12[1])
        l_13_1_isnan = np.isnan(l_13[1])
        l_24_1_isnan = np.isnan(l_24[1])
        l_34_1_isnan = np.isnan(l_34[1])

        #Checks both lines are non-horizontal
        if l_12_1_isnan == False and l_13_1_isnan == False:
            
            #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given smallest and second smallest j values respectively
            for j in range(sorted_rhombus[0, 1], sorted_rhombus[1, 1] + 1):
                
                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_12
                #The upper range of i is the x value at the given y value for the line l_13
                for i in range(round((l_12[0] * (j - p_1[1])) + p_1[0]), round((l_13[0] * (j - p_1[1])) + p_1[0]) + 1):

                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1

        #Checks if line l_12 is horizontal
        elif l_12_1_isnan == True and l_13_1_isnan == False:

            #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given smallest and second lowest j values respectively
            for j in range(sorted_rhombus[0, 1], sorted_rhombus[1, 1] + 1):

                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_12 (as the line is horizontal, this is a static value)
                #The upper range of i is the x value at the given y value for the line l_13
                for i in range(p_1[0], round((l_13[0] * (j - p_1[1])) + p_1[0]) + 1):

                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1
        
        #Checks if line l_13 is horizontal
        elif l_12_1_isnan == False and l_13_1_isnan == True:

            #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given  smallest and second smallest j values respectively
            for j in range(sorted_rhombus[0, 1], sorted_rhombus[1, 1] + 1):

                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_12 (as the line is horizontal, this is a static value)
                #The upper range of i is the x value at the given y value for the line l_13
                for i in range(round((l_12[0] * (j - p_1[1])) + p_1[0]), p_1[0] + 1):

                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1

        #Checks which point p_2 and p_3 have the lowest j value
        if p_2[1] < p_3[1]:
            
            #Checks both lines are non-horizontal
            if l_24_1_isnan == False and l_13_1_isnan == False:

                #A loop going from the top of the array down (increasing j)
                #The lower range and upper range are the given second smallest and second largest j values respectively
                for j in range(sorted_rhombus[1, 1], sorted_rhombus[2, 1]):

                    #A loop going from the left to the right of the array (increasing i)
                    #The lower range of i is the x value at the given y value for the line l_24
                    #The upper range of i is the x value at the given y value for the line l_13
                    for i in range(round((l_24[0] * (j - p_2[1])) + p_2[0]), round((l_13[0] * (j - p_1[1])) + p_1[0]) + 1):

                        #Attributes to the array coordinates within the rhombus the value of 1
                        rhombus_array[j, i] = 1
                
            #Checks if lines are horizontal
            else:

                #A loop going from the top of the array down (increasing j)
                #The lower range and upper range are the given second smallest and second largest j values respectively
                for j in range(sorted_rhombus[1, 1], sorted_rhombus[2, 1]):

                    #A loop going from the left to the right of the array (increasing i)
                    #The lower range of i is the x value at the given y value for the line l_24 (which as horizontal is static)
                    #The upper range of i is the x value at the given y value for the line l_13 (which as horizontal is static)
                    for i in range(p_2[0], p_1[0] + 1):

                        #Attributes to the array coordinates within the rhombus the value of 1
                        rhombus_array[j, i] = 1
        else:
            
            #Checks both lines are non-horizontal
            if l_12_1_isnan == False and l_34_1_isnan == False:

                #A loop going from the top of the array down (increasing j)
                #The lower range and upper range are the given second smallest and second largest j values respectively
                for j in range(sorted_rhombus[1, 1], sorted_rhombus[2, 1]):

                    #A loop going from the left to the right of the array (increasing i)
                    #The lower range of i is the x value at the given y value for the line l_12
                    #The upper range of i is the x value at the given y value for the line l_34
                    for i in range(round((l_12[0] * (j - p_1[1])) + p_1[0]), round((l_34[0] * (j - p_3[1])) + p_3[0]) + 1):

                        #Attributes to the array coordinates within the rhombus the value of 1
                        rhombus_array[j, i] = 1

            #Checks if lines are horizontal
            else:

                #A loop going from the top of the array down (increasing j)
                #The lower range and upper range are the given second smallest and second largest j values respectively
                for j in range(sorted_rhombus[1, 1], sorted_rhombus[2, 1]):

                    #A loop going from the left to the right of the array (increasing i)
                    #The lower range of i is the x value at the given y value for the line l_12 (which as horizontal is static)
                    #The upper range of i is the x value at the given y value for the line l_34 (which as horizontal is static)
                    for i in range(p_1[0], p_3[0] + 1):

                        #Attributes to the array coordinates within the rhombus the value of 1
                        rhombus_array[j, i] = 1

        #Checks both lines are non-horizontal
        if l_24_1_isnan == False and l_34_1_isnan == False:

            #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given second largest and largest j values respectively
            for j in range(sorted_rhombus[2, 1], sorted_rhombus[3, 1] + 1):

                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_24
                #The upper range of i is the x value at the given y value for the line l_34
                for i in range(round((l_24[0] * (j - p_2[1])) + p_2[0]), round((l_34[0] * (j - p_3[1])) + p_3[0]) + 1):

                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1
        
        #Checks if line l_24 is horizontal
        elif l_24_1_isnan == True and l_34_1_isnan == False:

            #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given second largest and largest j values respectively
            for j in range(sorted_rhombus[2, 1], sorted_rhombus[3, 1] + 1):

                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_24 (which is static)
                #The upper range of i is the x value at the given y value for the line l_34
                for i in range(p_2[0], round((l_34[0] * (j - p_3[1])) + p_3[0]) + 1):
                    
                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1

        #Checks if line l_34 is horizontal
        elif l_24_1_isnan == False and l_34_1_isnan == True:

             #A loop going from the top of the array down (increasing j)
            #The lower range and upper range are the given second largest and largest j values respectively
            for i in range(sorted_rhombus[2, 1], sorted_rhombus[3, 1] + 1):

                #A loop going from the left to the right of the array (increasing i)
                #The lower range of i is the x value at the given y value for the line l_24
                #The upper range of i is the x value at the given y value for the line l_34 (which is static)
                for i in range(round((l_24[0] * (j - p_2[1])) + p_2[0]), p_3[0] + 1):

                    #Attributes to the array coordinates within the rhombus the value of 1
                    rhombus_array[j, i] = 1
        
        rhombus_data = {
            "vertices": sorted_rhombus,
            "array": rhombus_array
        }

        return rhombus_data

class Analyser():
    """
    A class which applies changes to a set of vertices and/or array describing a rhombus

    Attributes
    ----------
    filename : PathLike
        A bmp file to use as comparison to the afected vertices
    image_array : ndarray
        An ndarray describing the image usaed to initialise tha class
    A_total : int
        The area of the image used to initialise the class

    Methods
    -------
    LCalc()
        1Calculates a value, L, comparing the area and density of a generated array compared to the example array provided
    rotation()
        Applies a rotation matrix of random angle, theta, within +- pi/10 to a set of vertices with centre image_rombus_midpoint
    diagonal_change()
        Changes the distant between two opposite vertices of a rhombus chosen randomly within +- 20 pixels distance
    transform()
        Generates a scatterlot of an array
    """

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename : PathLike
            A bmp file to use as comparison to the afected vertices
        """

        image = ImageProcessor(convert_bmp_to_array(filename))
        self.image_array = image.clean_image(5)
        self.A_total = 1000 * 600

    def LCalc(self, rhombus_array: dict):
        """Calculates a value, L, comparing the area and density of a generated array compared to the example array provided
        
        Parameters
        ----------
        rhombus_array : ndarray
            The generated array that will be modified
        """

        #Sets parameter alpha to 3
        alpha = 3

        #Area of the generated rhombus in pixels
        A = np.count_nonzero(rhombus_array["array"])

        #Multiply two array's element wise to calculate area of overlap
        sigma = np.count_nonzero(rhombus_array["array"] * self.image_array)
        return -(alpha * (A / self.A_total) + sigma / A)
    
    def rotation(self, current_rhombus: dict):
        """Applies a rotation matrix of random angle, theta, within +- pi/10 to a set of vertices with centre image_rombus_midpoint
        
        Parameters
        ----------
        current_rhombus : ndarray
            The set of vertices that the rotation is applied to
        """

        #Generates random angle theta between +- pi/10 radians
        theta = (np.random.rand() - 0.5) * (np.pi / 5)

        #Creates the rotation matrix with angle theta
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        candidate_vertices = np.zeros((4, 2))
        
        #Loop for number of vertices
        for i in range(0, 4):

            #Rotates the verticex about the centre
            candidate_vertices[i, 0] = image_rhombus_midpoint[0] + np.matmul(rotation_matrix, np.reshape((current_rhombus["vertices"][i] - image_rhombus_midpoint), (2, 1)))[0]
            candidate_vertices[i, 1] = image_rhombus_midpoint[1] + np.matmul(rotation_matrix, np.reshape((current_rhombus["vertices"][i] - image_rhombus_midpoint), (2, 1)))[1]

        candidate_data = Rhombus.array_generator(candidate_vertices)
        
        return candidate_data
    
    def diagonal_change(self, current_rhombus: dict):
        """Changes the distant between two opposite vertices of a rhombus chosen randomly within +- 20 pixels distance

        Parameters
        ----------
        current_rhombus : ndarray
            The set of vertices that the diagonal change is applied to
        """

        choice1 = np.random.randint(0, 2)

        #Generates random scalar between +- 10
        delta = (np.random.rand() - 0.5) * 20
        candidate_vertices = np.zeros((4, 2))

        if choice1 == 0:

            #Calculates unit vector parallel to line l_14
            l_14 = (current_rhombus["vertices"][3] - current_rhombus["vertices"][0]) / (np.linalg.norm(current_rhombus["vertices"][3] - current_rhombus["vertices"][0]))

            #Checks l_14 isn't horizontal
            if np.isnan(l_14[1]) == False:
                candidate_vertices[1] = current_rhombus["vertices"][1]
                candidate_vertices[2] = current_rhombus["vertices"][2]

                #p_1 and p_4 are moved along the l_14 delta pixels each
                candidate_vertices[0] = current_rhombus["vertices"][0] - (l_14 * delta)
                candidate_vertices[3] = current_rhombus["vertices"][3] + (l_14 * delta)
            
            else:
                candidate_vertices[1] = current_rhombus["vertices"][1]
                candidate_vertices[2] = current_rhombus["vertices"][2]

                #p_1 and p_4 are moved along the l_14 delta pixels each
                candidate_vertices[0, 0] = current_rhombus["vertices"][0, 0] - delta
                candidate_vertices[3, 0] = current_rhombus["vertices"][3, 0] + delta
        
        else:

            #Calculates unit vector parallel to line l_14
            l_23 = (current_rhombus["vertices"][2] - current_rhombus["vertices"][1]) / (np.linalg.norm(current_rhombus["vertices"][2] - current_rhombus["vertices"][1]))

            #Checks l_23 isn't horizontal
            if np.isnan(l_23[1]) == False:
                candidate_vertices[0] = current_rhombus["vertices"][0]
                candidate_vertices[3] = current_rhombus["vertices"][3]

                #p_2 and p_3 are moved along the l_23 delta pixels each
                candidate_vertices[1] = current_rhombus["vertices"][1] - (l_23 * delta)
                candidate_vertices[2] = current_rhombus["vertices"][2] + (l_23 * delta)

            else:
                candidate_vertices[0] = current_rhombus["vertices"][0]
                candidate_vertices[3] = current_rhombus["vertices"][3]

                #p_2 and p_3 are moved along the l_23 delta pixels each
                candidate_vertices[1, 0] = current_rhombus["vertices"][1, 0] - delta
                candidate_vertices[2, 0] = current_rhombus["vertices"][2, 0] + delta
            
        candidate_data = Rhombus.array_generator(candidate_vertices)
        
        return candidate_data
    
    def transform(self, array):
        """Generates a scatterlot of an array

        Parameters
        ----------
        array : ndarray
            The array to generate a scatterplot from 
        """

        plt.figure(figsize = [10, 6])
        plt.xlim(0, 1000)
        plt.ylim(0, 600)
        for i in range(0, 1000, 5):
            for j in range(0, 600, 5):
                if array[j, i] == 1:
                    plt.scatter(i, j, s = 2, c = 'b')

initial_vertices = Rhombus.random_vertices_generator()
working_rhombus = Rhombus.array_generator(initial_vertices)
T = 5
L_bar_new = 1
analyser = Analyser("C:\\Users\\sambi\\Downloads\\Image2.bmp")
L_opt = analyser.LCalc(working_rhombus)
L_data = np.array([])
xvalues = np.array([])

for x in range(0, 100):
    sum = 0

    for y in range(0, 1000):

        choice2 = np.random.randint(0, 2)
        r = np.random.rand()

        if choice2 == 0:
            candidate_rhombus = analyser.rotation(working_rhombus)
        
        else:
            candidate_rhombus = analyser.diagonal_change(working_rhombus)

        #Calculates the change in L beween the candidate rhombus and the working rhombus
        Delta_L = analyser.LCalc(candidate_rhombus) - L_opt

        #Checks if Delta_L is negative and if so accept candidate rhombus
        if Delta_L <= 0:
            working_rhombus = candidate_rhombus
            L_opt = Delta_L + L_opt
        
        #Extra condition to accept candidate rhombus
        elif np.exp(-(Delta_L / T)) < r:
            working_rhombus = candidate_rhombus
            L_opt = Delta_L + L_opt
        
        L_data = np.append(L_data, L_opt)
        xvalues = np.append(xvalues, y + 1000 * x)
        #Adds all values of L for the accepted rhombuses
        sum += L_opt
        
    L_bar_old = L_bar_new

    #Calculates new average L over past 1000 runs
    L_bar_new = sum / 1000
    percent_change_L = (L_bar_new - L_bar_old) / L_bar_old

    #When percentage change is less than 0.25% stops the loops and returns the array
    if abs(percent_change_L) <= 0.0025:
        break

plt.plot(xvalues, L_data)
plt.xlabel("Iteration number")
plt.ylabel("L")
plt.savefig("C:\\Users\\sambi\\Pictures\\Computing Challenge 2023_24\\L_against_iteration_image2 - Coding Group 31.jpeg")
plt.show()

final_rhombus = working_rhombus["array"]
analyser.transform(final_rhombus)
plt.imshow(analyser.image_array, cmap="Blues", alpha = 0.2)
plt.savefig("C:\\Users\\sambi\\Pictures\\Computing Challenge 2023_24\\Scatterplot_comparison_image2 - Coding Group 31.jpeg")
plt.show()

fig, ax = plt.subplots()
ax.imshow(final_rhombus, cmap = "Greys")
ax.imshow(analyser.image_array, cmap="Blues", alpha = 0.2)
plt.savefig("C:\\Users\\sambi\\Pictures\\Computing Challenge 2023_24\\imshow_comparison_image2 - Coding Group 31.jpeg")
plt.show()

print(np.count_nonzero(final_rhombus))