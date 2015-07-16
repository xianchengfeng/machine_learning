# -*- coding: utf-8 -*-
from numpy import *
def loadDataSet(filename):
	dataMat = []; labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat
def selectJrand(i,m):    #i是第一alpha的下标，m是所有alpha的数目，只要函数值不等于输入值i，就会随机选择
	j = i
	while(j == i):
		j = int(random.uniform(0,m))
	return j 
def clipAlpha(aj,H,L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
	dataMat = mat(dataMatIn); labelMat = mat(classLabels).transpose()
	b = 0; m,n = shape(dataMat)
	alphas = mat(zeros((m,1)))
	iter = 0
	while (iter < maxIter):
		alphaPairChanged = 0
		for i in range(m):
			fXi = float(multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T)) + b
			Ei = fXi - float(labelMat[i])
			if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
				j = selectJrand(i,m)
				fXj = float(multiply(alphas,labelMat).T*(dataMat*dataMat[j,:].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIold = alphas[i].copy();alphaJold = alphas[j].copy()
				if (labelMat[i]!=labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C+ alphas[j]-alphas[i])
				else:
					L = max(0, alphas[j] +alphas[i] -C)
					H = min(C, alphas[j] + alphas[i])
				if L==H: print("L==H"); continue
				eta = 2.0*dataMat[i,:]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T -\
					dataMat[j,:]*dataMat[j,:].T
				if eta >= 0 :print("eta>=0");continue
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				alphas[j] = clipAlpha(alphas[j],H,L)
				if(abs(alphas[j]-alphaJold) < 0.00001): print("j no moving enough"); continue
				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*\
					dataMat[i,:]*dataMat[i,:].T - \
					labelMat[j]*(alphas[j]-alphaJold)*\
					dataMat[i,:]*dataMat[j,:].T 
				b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*\
					dataMat[i,:]*dataMat[j,:].T - \
					labelMat[j]*(alphas[j]-alphaJold)*\
					dataMat[j,:]*dataMat[j,:].T 
				if (0 < alphas[i]) and (C > alphas[j]):	b=b1
				elif (0<alphas[j]) and (C > alphas[j]):	b=b2
				else: b = (b1 + b2)/2.0
				alphaPairChanged += 1
				print("iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairChanged))
		if (alphaPairChanged == 0): iter += 1
		else:	iter = 0
		print("iteration number: %d" % iter)
	return b,alphas
class optStruct:
	def __init__(self,dataMatIn, classLabels, C, toler): 
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))
def calcEk(oS, k):
	fXk = float(multiply(oS.alphas,oS.labelMat).T*\
	(oS.X*oS.X[k,:].T)) + oS.b
	Ek = fXk - float(oS.labelMat[k])
	return Ek
def selectJ(i, oS, Ei): 
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1,Ei] 
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]
	if (len(validEcacheList)) > 1:
		for k in validEcacheList: 
			if k == i: continue 
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			if (deltaE > maxDeltaE): 
				maxK = k; maxDeltaE = deltaE; Ej = Ek 
		return maxK, Ej
	else: 
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej
def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1,Ek]