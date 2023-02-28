## Download iris data set on an EC2 instance and compare the accuracy of KNN,SVM, and ANN.

* Go to the AWS Management Console (https://aws.amazon.com/console/). You can log in to your AWS Account. On services, tab search for EC2 and click on instances in the sidebar for creating an EC2 Instance.
* For creating a new EC2 instance click on launch instance.
* First, enter a name tag for the instance. This will act as key-value pair and if you wanted to add additional tags to tag your instance differently, then you could click on “Add additional tags”.
* Next, you need to choose a base image for your EC2 instance i.e. an operating system of your instance. You can see a full catalog of OS that you can search but in this, we’re going to use Amazon Linux, which is provided by AWS. Now you can select any OS image for your EC2 instance which will run in your EC2 instance.
* Next, we need to choose an instance type. Instance types are going to differ based on the number of CPUs they have, the amount of memory they have, and how much they cost. In this article, we are going to have a T2 micro selected. This one is free tier eligible, so it will be free to launch them. If you wanted to compare the instance types click on “Compare Instance types” it shows you all the types of instances as well as how much memory they have and so on.
* Next, create a key pair to log into your instance. So this is necessary if we use the SSH utility to access our instance. I have already created a key pair and downloaded it to my computer which I can use during SSH.
* For creating key-value pair we can use these steps:
    * Give a name to Key Pair. Then you need to choose a key pair type (RSA encrypted or ED25519 encrypted), here we’ll be using the RSA encrypted.
    * And then we have to select key pair formats. If you have Mac, Linux, or Windows 10, you can use the .pem format.
    * If you have Windows less than version 10, for example, Windows 7 or Windows 8, then you can use a .ppk format.
    * When you create a key pair it will be downloaded automatically to your pc.
* Next, we have to go into network settings. Leave the settings to default Our instances are going to get a public IP and then there is going to be a security group attached to our instance (which is going to control the traffic from and to our instance ) and therefore we can add rules. The first security group created will be called launch-wizard-1 which is created by the console directly and we can define multiple rules here. So the first rule we want to have is to allow SSH traffic from anywhere and to allow HTTP traffic from the internet.
* Next, we need to configure the storage for this instance let’s have an eight-gigabyte gp2 root volume because, in the free tier, we can get up to 30 gigabytes of EBS General Purpose SSD storage. If you go into advanced details you could configure them a little more advanced in terms of storage, volume, encryption, and keys. One important thing to note here is the “delete on termination’ should be “Yes”. By default, it is enabled to yes, which means that once we terminate our EC2 instance, then this volume is also going to be deleted.Add 30GB storage so that we can download different python packages.
* User Data Script: 
    * User data is when we pass a script with some commands to our EC2 instance. It will execute the script on the first launch of our EC2 instance and only on the first launch. And therefore, on the first launch, we shall be able to pass the commands here. I have not given any data script.
    * Finally, we can review everything we have created and launch this instance.

Once the instance name is created we can observe a few things
There’s an instance ID which is just a unique identifier for our instance.
There is a public IPv4 address, this is what we’re going to use to access our EC2 instance,
There is a private IPv4 address which is how to access that instance internally on the AWS network, which is
private.
The current state of the instance. The instance state is running,
We also get some information about hostname, private DNS, Instance type, AMI details, and Key pair.
Click on “Connect to instance“.

#### Now you can update the OS and install what you need on this EC2 instance.

Install Pyhon packages necessarily for this project like pandas, numpy, scikit learn and tensorflow.

* git clone this repository 
* go inside this repository and run the KNN, SVM and ANN to know the accuracy.