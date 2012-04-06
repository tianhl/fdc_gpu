ROOTCONFIG   := root-config
ROOTCFLAGS   := $(shell $(ROOTCONFIG) --cflags)
ROOTLDFLAGS  := $(shell $(ROOTCONFIG) --ldflags)
ROOTGLIBS    := $(shell $(ROOTCONFIG) --glibs)

CXX           = g++
CXXFLAGS      = -O2 -Wall -fPIC
LD            = g++
LDFLAGS       = -O2
SOFLAGS       = -shared

CXXFLAGS     += $(ROOTCFLAGS)
LDFLAGS      += $(ROOTLDFLAGS)
LIBS          = $(ROOTLIBS) $(SYSLIBS)
GLIBS         = $(ROOTGLIBS) $(SYSLIBS)

CXXFLAGS     += -I$(ATISTREAMSDKROOT)/include
LDFLAGS      += -L$(ATISTREAMSDKROOT)/lib/x86_64 -lOpenCL

OBJS          = test.o OpenPWA.o config.o log.o 

test : $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS) $(GLIBS)

$(OBJS) : %.o: %.cxx
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY : clean

clean :
	@rm $(OBJS) test
	@echo "Cleanning everything ... " 
