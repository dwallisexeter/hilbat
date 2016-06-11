This file is part of Hilbat.

    Hilbat is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hilbat is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import pysoundfile as sf
import pickle
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import logging
import batplot

markers=[1760362, 1772512, 1787762, 1799712, 1814912, 1829662, 1858334]
#markers=[1760362, 1772512]#, 1787762, 1799712, 1814912, 1829662, 1858334]

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def batang(phi,v=330.0,f=40e3,d=0.0127,phi_cal=1.77,wrap=0):
    a=(phi-phi_cal+(2*np.pi*wrap))*v/(2*np.pi*d*f)
    a[a>1.0]=1.0
    a[a<-1.0]=-1.0
    return np.arcsin(a)

def batangdeg(phi,v=330.0,f=40e3,d=0.0127,phi_cal=1.77,wrap=0):
    logging.info("phi_cal:%e" % phi_cal)
    return batang(phi,v=v,f=f,d=d,phi_cal=phi_cal,wrap=wrap)/2.0/np.pi*360

def straightline(x, A, B):
    return A*x + B

def wmstd(a,w):
    N=a.size*1.0
    wm=np.sum(a*w)/np.sum(w)
    d=(a-wm)**2
    top=np.sum(w*d)
    top=np.abs(top)
    bot=(N-1)/N * np.sum(w)
    bot=np.abs(bot)
    return (wm,np.sqrt( top/bot  ))

class BatCall(object):
    def __init__(self,hb,position,L,ifprops):
        self.hb=hb
        self.wav=self.hb.wav
        self.position=position
        self.centre=0
        self.winWidth=0
        self.L=L
        self.wrap=0
        self.guess=True
        self.ifprops=ifprops
#        self.xtime=self.hb.xtime

    def _readdata(self,pos,span):
        start=int(pos-span/2)
        self.wav.seek_absolute(start)
        return self.wav.read(span)
        
    def read(self,pos=0,span=0):
        if pos==0:
            pos=self.position
        if span==0:
            span=self.L
        d=self._readdata(pos,span)
        self.dataA=d[:,0]
        self.dataB=d[:,1]

    def _convTime(self,x):
        sampleRate=self.wav.sample_rate
        x=x/sampleRate*1e3
        return x

    def time(self):
        return self._convTime(self.centre)

    def getX(self,pos=0,span=0):
        if pos==0:
            pos=self.position
        if span==0:
            span=self.L
        start=int(pos-span/2)
        x=np.arange(start,start+span)*1.0
        if self.hb.xtime:
            x=self._convTime(x)
        return x

    def make(self):
        self.read()
        self.doHilbert()
        self.doBatAngle()
        self.read()

    def doHilbert(self):
        self.hA=hilbert(self.dataA)
        self.hB=hilbert(self.dataB)
        self.AA=np.absolute(self.hA)
        self.AB=np.absolute(self.hB)
        self.phiA=np.angle(self.hA)
        self.phiB=np.angle(self.hB)
        self.uwphiA=np.unwrap(self.phiA)
        self.uwphiB=np.unwrap(self.phiB)
        sr=self.wav.sample_rate
        self.fA=sr/(2*np.pi)*np.diff(self.uwphiA)
        self.fB=sr/(2*np.pi)*np.diff(self.uwphiB)
        phi_cal=self.ifprops['phiCal']
        uw=np.angle(self.hB)-np.angle(self.hA)-phi_cal
        uw[uw<-np.pi]+=2*np.pi
        uw[uw>=np.pi]-=2*np.pi
        self.phi=uw

    def doBatAngle(self):
        phiCal=self.ifprops['phiCal']
        micSpacing=self.ifprops['micSpacing']
        sos=self.ifprops['sos']
        if self.guess:
            self.guessWindow()
        print self.centre,self.winWidth
        self.read(self.centre,self.winWidth)
#        amp,mean,std=self.windowStats()
        mid=-self.position+self.centre+self.L/2
        st=mid-self.winWidth
        end=mid+self.winWidth
        print "start,end",st,end
        x=self.getX()
        # Bat Angle
        uw=np.angle(self.hB)-np.angle(self.hA)-phiCal
        uw[uw<-np.pi]+=2*np.pi
        uw[uw>=np.pi]-=2*np.pi
        phase=np.unwrap(uw[st:end])
        frequency=self.fA[st:end]
        wrap=self.wrap
        print "WRAP",wrap
        self.batangle=batangdeg(phase,f=frequency,v=sos,
            d=micSpacing/1000.0,phi_cal=phiCal,wrap=wrap)
        self.batanglex=x[st:end]
        adiff=np.abs(self.hA[st:end] - self.hB[st:end])
        aprod=np.abs(self.hA[st:end] * self.hB[st:end])
        ad=aprod
        wt=ad/np.max(ad)

        self.meanBatAngle,self.stdBatAngle=wmstd(self.batangle,wt)

    def guessWindow(self):
        self.read()
        amp,mean,std=self.windowStats()
        self.centre=int(mean)
        self.winWidth=int(std)
        print "guessWindow()", self.centre,self.winWidth

    def windowStats(self):
        guesses=[1,self.position,100]
        x=self.getX()
        popt0,pcov0 = curve_fit(gaus,x,self.AA,p0=guesses)
        popt1,pcov0 = curve_fit(gaus,x,self.AB,p0=guesses)
        popt=(np.abs(popt0)+np.abs(popt1))/2.0
        return popt[0],popt[1],popt[2]
        
    def gaussWindow(self):
        x=self.getX()        
        amp,mean,std=self.windowStats()
        return gaus(x,amp,mean,std)
        
    def gaussBoundaries(self):
        x=self.getX()        
        amp,mean,std=self.windowStats()
        return mean-std,mean+std 


class HilBat(object):
    def __init__(self,fname=None):
        self._initInit()
        if fname:
            self.openWav(fname)

    def _initInit(self):
        # Initialise things
        self.fileName=''
        self.wav=None
        self.wavfilename=''
        self.ifprops={'micSpacing': 12.75, 'phiCal': 1.77,'sos': 330}
        self.calls=[]
#        self.L=1024
        self.callNumber=-1
        self.pointer=0
        self.span=1024
        self.xtime=False
        self.fftWidth=256

    # I/O ##########################################
    def openWav(self,fname):
        self.wav=sf.SoundFile(str(fname))
        self.wavfilename=fname

    def _mark(self,marker):
        self.addBatCall(marker)
#        self.call().wrap=wrap
        self.call().make()

    def loadFile(self,fname):
        with open(fname) as f:
            d=pickle.load(f)
            print d
            fn,markers,marker,markeri,phi_cal,micSpacing,sos,wrapArray=d
            self.openWav(fn)
            self.wavfilename=fn
            for i in range(len(markers)):
                self._mark(markers[i])#,wrapArray[i])
                self.call().wrap=wrapArray[i]
            self.ifprops={'micSpacing': micSpacing, 
                          'phiCal': phi_cal,
                          'sos': sos}
            self.gotoCall(markeri)
            self.fileName=fname

    def saveFile(self,fname):
        wrapArray=[c.wrap for c in self.calls]
        markers=[c.position for c in self.calls]
        d=[self.wavfilename,
           markers,
           0,
           self.callNumber,
           self.ifprops['phiCal'],
           self.ifprops['micSpacing'],
           self.ifprops['sos'],
           wrapArray]
        with open(fname,'wb') as f:
            pickle.dump(d,f)
        self.fileName=fname

    def frames(self):
        return self.wav.frames

    def setPointer(self,i):
        self.pointer=i
        m=self.frames()-self.span
        if self.pointer>m:
            self.pointer=m
        if self.pointer<0:
            self.pointer=0
        self.read()
        self.doSonoGraph2()
    
    def setSpan(self,i):
        self.span=i
        self.read()
        self.doSonoGraph2()

    def _readdata(self):
        start=self.pointer-self.span/2
        start=self.pointer
        self.wav.seek_absolute(start)
        return self.wav.read(self.span)
        
    def read(self):
        d=self._readdata()
        self.dataA=d[:,0]
        self.dataB=d[:,1]


    def _convTime(self,x):
        sampleRate=self.wav.sample_rate
        x=x/sampleRate*1e3
        return x

    def getX(self):
        L=self.span
        start=self.pointer#-self.span/2
        x=np.arange(start,start+L)*1.0
        if self.xtime:
            x=self._convTime(x)
        return x

    def _hann(self,N):
        n=np.arange(N)
        return 0.5*(1-np.cos(2*np.pi*n/(N-1)))

    def doSonoGraph(self):
        self.read()
        A=self.dataA*1.0
        hann=self._hann(A.size)
        A=A*hann
        Wf=self.fftWidth
        Wd=self.span
        nbins=Wd/Wf
        a=np.zeros((Wf,nbins),dtype=complex)
        for i in range(nbins):
            start=i*Wf
            end=start+Wf
            a[:,i]=np.fft.fft(A[start:end])
        self.sono=a
        si=1.0/self.wav.sample_rate
        print "si",si
        self.sonofreq=np.fft.fftfreq(Wf,d=si)
        
    def doSonoGraph2(self):
        self.read()
        A=self.dataA*1.0
#        A=A*hann
        Wf=self.fftWidth
        Wd=self.span
        a=np.zeros((Wf,Wd),dtype=complex)
        hann=self._hann(Wf)
        for i in range(Wd-Wf):
            start=i
            end=start+Wf
            c=A[start:end]
            c=c*hann
            a[:,i]=np.fft.fft(c)
        self.sono=a
        si=1.0/self.wav.sample_rate
        print "si",si
        self.sonofreq=np.fft.fftfreq(Wf,d=si)
        


    # CALLS ########################################
    def makeCall(self,position):
        b=BatCall(self,position,self.span,self.ifprops)
        b.make()
        return b

    def addBatCall(self,position):
        self.calls.append(self.makeCall(position))
        self.callNumber=len(self.calls)-1

    def ncalls(self):
        return len(self.calls)

    def call(self):
        if len(self.calls)>0:
            c=self.calls[self.callNumber]
#            c.make()
            return c
        else:
            return None

    def incCall(self):
        self.callNumber=self.callNumber+1
        if self.callNumber>self.ncalls()-1:
            self.callNumber=self.ncalls()-1
        c=self.calls[self.callNumber]
        c.make()

    def decCall(self):
        self.callNumber=self.callNumber-1
        if self.callNumber<0:
            self.callNumber=0
        c=self.calls[self.callNumber]
        c.make()

    def gotoCall(self,n):
        if len(self.calls)==0:
            logging.info("No Calls")
            return
        self.callNumber=n
        if self.callNumber>self.ncalls()-1:
            self.callNumber=self.ncalls()-1
        if self.callNumber<0:
            self.callNumber=0
        c=self.calls[self.callNumber]
        c.make()

    def removeCurrent(self):
        self.calls.pop(self.callNumber)
#        self.calls.remove([self.call()])
        self.gotoCall(self.callNumber)
        
def test():
    h=HilBat('be2.wav')
    h.addBatCall(markers[0])
    h.calls[0].make()
#    p=BatCallPlot(h.calls[0],figure=plt.figure(2))
    return h,p

import matplotlib.pyplot as plt

def test2():
    h=HilBat('be2.wav')
    h.addBatCall(markers[0])
    h.call().make()
    p=batplot.BatPlot(h,figure=plt.figure(2))
    return h,p

def test3():
    h=HilBat('be2.wav')
    h.addBatCall(markers[0])
    h.call().make()
    p=batplot.BatPlot(h,figure=plt.figure(2))
    return h,p
