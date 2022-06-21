#�޶�ʱ�䣺2019��11��19��
# ȥ������Ҫ�Ķ���
#
#

#�޶�ʱ��: 2008��7��28��

# Synopsis:   compute complexity, cohesion, coupling, inheritance, and size metrics for each class in project
#
# Language: C++, Java
#
# Description:
#  ������ĸ�����/�ھ���/�����/��ģ/�̳е���ض��� 
#  For the latest Understand perl API documentation, see 
#      http://www.scitools.com/perl/
#  Refer to the documenation for details on using the perl API 
#  with Understand and for details on all Understand perl API calls.

# �ٶȿ�İ汾: (1) ����Զ������������������, �ܵĸ�����ΪO(N), NΪ���ݿ��������Ŀ
#               (2) �̳���صĶ���������һ���ӳ����м���, ֻ��Ҫ��һ��������/������, �����Զ���������ǰ�ļ�����
# ��Ҫ����: �ṹ��������, �����ڸ��õ�����Ӧ���� 

use strict;
use Understand;
use Getopt::Long;
use File::Basename;
use IO::File;
use POSIX qw(tmpnam);
use Env;
use File::Find;

autoflush STDOUT 1;

# Usage:
sub usage
{
    return << "END_USAGE";
${ \( shift @_ ) }
usage: computeChangeProness -prev database -out file
 -prev database             Specify Understand database (required for uperl, inherited from Understand)
 -out file                  (optional)Output file, default is "D:/test.csv" 
END_USAGE
}

my %opts=( 
    db => "D:/workspace/mixed-workspace/mySZZ/GetMetrics/udb/kafka/kafka-2.1.1.udb",
    out => 'D:/workspace/mixed-workspace/mySZZ/GetMetrics/metrics/kafka/test.csv',
    help => '',
    );

# my $dbPath;
# my $comma;
# my $help;
GetOptions(
     "db=s" => \$opts{db},
     "out=s" => \$opts{out},
     "help" => \$opts{help},
           );

use FileHandle;
my $outputFileName = $opts{out};
my $outputFH = new FileHandle(">  ".$outputFileName);

# help message
die usage("") if ($opts{help});
die usage("database specification required") if (!$opts{db});

# insist the it not be run from within the GUI
if ( Understand::Gui::active() ) {
    die usage("This script is not designed to be called from the GUI");
}

# make sure db files are readable before continuing on
die "Can't read database " . $opts{'db'} if ( $opts{db} && ! -r $opts{db} );

# open the database 
my $db=openDatabase($opts{db});

#debug options
my $debug = 0;

#ȫ�ֱ���
my %allClassNameHash; #���ݿ������е���, keyΪ����




# verify supported language
my $language = $db->language();
if ($language !~ /ada|c|java/i) {
    closeDatabase($db);
    die "$language is currently unsupported";
}


my $sep = ",";
my $totalClasses = 0;

print "\nhi, I am computing ",$opts{db}, " please wait ... \n";

my ($isConstructor, $isDestructor, $isAccessOrDelegationMethod) = initialSpecialMethod($language);

# get sorted class entities list
#ֻ������������, ����ͳ����ЩǶ�����������е���
my @tempclasses = $db->ents("class ~unknown ~unresolved");
my @classes;

print "total = ", scalar @tempclasses, "\n";


foreach my $class (@tempclasses){
	 print "standar class = ", $class->name(), "\n" if ($class->library() =~ m/Standard/i);
   next if ($class->library() =~ m/Standard/i);
   
	 next if ($class->ref("definein", "class"));
	 next if ($class->ref("definein", "interface"));
	 next if ($class->ref("definein", "method"));     # Java�ķ������ܻᶨ����������(ͨ�������ڷ��������)
	 next if ($class->ref("definein", "function"));
	 
	 my ($startRef) = $class->refs("definein", "", 1); #understand�п��ܽ�����û�ж�������������, ԭ��δ֪
	 my ($endRef) = $class->refs("end","",1);		
	 next if (!$startRef || !$endRef);
	   
	 push @classes, $class;
	 
	 my $classKey = getClassKey($class); 
	 $allClassNameHash{$classKey} = $class;  #��¼���ݿ������е���
}


my $metricsList = availableMetrics();

my @complexityMetricNameList = (sort keys %{$metricsList->{Complexity}});
my @sizeMetricNameList = (sort keys %{$metricsList->{Size}});
my @InheritanceMetricNameList = (sort keys %{$metricsList->{Inheritance}});
my @cohesionMetricNameList = (sort keys %{$metricsList->{Cohesion}});
my @couplingMetricNameList = (sort keys %{$metricsList->{Coupling}});
my @otherMetricNameList = (sort keys %{$metricsList->{Other}});

print $outputFH ("relName".$sep."className".$sep.join($sep, @complexityMetricNameList)
                                .$sep.join($sep, @sizeMetricNameList)
                                .$sep.join($sep, @InheritanceMetricNameList)
                                # .$sep.join($sep, @cohesionMetricNameList)
                                .$sep.join($sep, @couplingMetricNameList)
                                # .$sep.join($sep, @otherMetricNameList)
                                , "\n");
# print $outputFH ("name_id".$sep.join($sep, @complexityMetricNameList)
#                                 .$sep.join($sep, @sizeMetricNameList)
#                                 .$sep.join($sep, @InheritanceMetricNameList)
#                                 # .$sep.join($sep, @cohesionMetricNameList)
#                                 .$sep.join($sep, @couplingMetricNameList)
#                                 # .$sep.join($sep, @otherMetricNameList)
#                                 , "\n");


print "total classes = ", scalar @classes, "\n";
print "total in hash = ", scalar (keys %allClassNameHash), "\n";


#my $BriandCouplingMetrics = getBriandCouplingMetrics(\%allClassNameHash);  #��������Briand�������18������Զ���
my $BriandCouplingMetrics = {};


foreach my $class (sort {$a->longname() cmp $b->longname();}@classes) {
	$totalClasses++;   

  print "\nNo = ", $totalClasses, " / ", scalar @classes;
  # print "\t class = ", $class->name(), "\n"; 
	
	my $metricsList = availableMetrics();	   #��һ�����Ҫ
	computeMetrics($class, \%allClassNameHash, $BriandCouplingMetrics, $metricsList);	

	# print $outputFH ($class->ref()->file()->relname());
	print $outputFH ($class->ref()->file()->relname(), $sep);
	print $outputFH ($class->longname());
	
	foreach my $metric (sort keys %{$metricsList->{Complexity}}){
		print $outputFH ($sep, $metricsList->{Complexity}->{$metric});
	}

	foreach my $metric (sort keys %{$metricsList->{Size}}){
		print $outputFH ($sep, $metricsList->{Size}->{$metric});
	}

	foreach my $metric (sort keys %{$metricsList->{Inheritance}}){
		print $outputFH ($sep, $metricsList->{Inheritance}->{$metric});
	}
	
	foreach my $metric (sort keys %{$metricsList->{Cohesion}}){
		print $outputFH ($sep, $metricsList->{Cohesion}->{$metric});
	}

	foreach my $metric (sort keys %{$metricsList->{Coupling}}){
		print $outputFH ($sep, $metricsList->{Coupling}->{$metric});
	}

	foreach my $metric (sort keys %{$metricsList->{Other}}){
		print $outputFH ($sep, $metricsList->{Other}->{$metric});
	}
	
	print $outputFH ("\n");
}



# my $totalSLOC = $db->metric("CountLineCode");
# print $outputFH ("\n Total SLOC in this project:  ", $totalSLOC, "\n");


close($outputFH);
closeDatabase($db);



sub initialSpecialMethod{	
	my $sLanguage = shift;
	
	my ($isConstr, $isDestr, $isAccOrDel);
	
	if ($sLanguage =~ /c/i){
    #	print "language is C++! \n";
	  $isConstr = \&isCPlusPlusConstructor;
	  $isDestr = \&isCPlusPlusDestructor;
	  $isAccOrDel = \&isCPlusPlusAccessOrDelegationMethod;
  }
  elsif ($sLanguage =~ /java/i){
    #	print "language is Java! \n";	
	  $isConstr = \&isJavaConstructor;
	  $isDestr = \&isJavaDestructor;
	  $isAccOrDel = \&isJavaAccessOrDelegationMethod;
  }
  
  return ($isConstr, $isDestr, $isAccOrDel);
}


sub availableMetrics{
	my $hashMetrics = {};
	
  #complexity metrics
  $hashMetrics->{Complexity}->{CDE} = "undef";        #Class Definition Entropy
  $hashMetrics->{Complexity}->{CIE} = "undef";        #Class Implementation Entropy
  $hashMetrics->{Complexity}->{WMC} = "undef";        #Weighted Method Per Class    
  $hashMetrics->{Complexity}->{SDMC} = "undef";       #Standard Deviation Method Complexity
  $hashMetrics->{Complexity}->{AvgWMC} = "undef";     #Average Weight Method Complexity
  $hashMetrics->{Complexity}->{CCMax} = "undef";      #Maximum cyclomatic complexity of a single method of a class
  $hashMetrics->{Complexity}->{NTM} = "undef";        #Number of Trival Methods  
#  $hashMetrics->{Complexity}->{CC1} = "undef";        #Class Complexity one. From: Y.S. Lee, B.S. Liang, F.J. Wang. Some complexity metrics for OO programs based on information flow. 
#  $hashMetrics->{Complexity}->{CC2} = "undef";        #Class Complexity Two. From: Y.S. Lee, B.S. Liang, F.J. Wang. Some complexity metrics for OO programs based on information flow. 
#  $hashMetrics->{Complexity}->{CC3} = "undef";        #Class Complexity Three. From: K. Kim, Y. Shin, C. Wu. Complexity measures for OO program based on the entropy
  
                      
  #coupling metrics
  $hashMetrics->{Coupling}->{CBO} = "undef";          #Coupling Between Object              
  # $hashMetrics->{Coupling}->{RFC} = "undef";          #Response For a Class, ���������ķ���������ֱ�ӻ��߼�ӵ��õķ���
 # $hashMetrics->{Coupling}->{RFC1} = "undef";         #Response For a Class, ֻ����������ķ�����ֱ�ӵ��õķ���
  # $hashMetrics->{Coupling}->{MPC} = "undef";          #Message Passing Coupling
  $hashMetrics->{Coupling}->{DAC} = "undef";          #Data Abstraction Coupling: �������������������Ŀ
  $hashMetrics->{Coupling}->{DACquote} = "undef";     #Data Abstraction Coupling: ������������������Ŀ
  $hashMetrics->{Coupling}->{ICP} = "undef";          #Information-flow-based Coupling
  $hashMetrics->{Coupling}->{IHICP} = "undef";        #Information-flow-based inheritance Coupling
  $hashMetrics->{Coupling}->{NIHICP} = "undef";       #Information-flow-based non-inheritance Coupling
#  $hashMetrics->{Coupling}->{IFCAIC} = "undef";       #Inverse friends class-attribute interaction import coupling
#  $hashMetrics->{Coupling}->{ACAIC} = "undef";        #Ancestor classes class-attribute interaction import coupling
#  $hashMetrics->{Coupling}->{OCAIC} = "undef";        #Others class-attribute interaction import coupling
#  $hashMetrics->{Coupling}->{FCAEC} = "undef";        #Friends class-attribute interaction export coupling
#  $hashMetrics->{Coupling}->{DCAEC} = "undef";        #Descendents class class-attribute interaction export coupling
#  $hashMetrics->{Coupling}->{OCAEC} = "undef";        #Others class-attribute interaction export coupling
#  $hashMetrics->{Coupling}->{IFCMIC} = "undef";       #Inverse friends class-method interaction import coupling
#  $hashMetrics->{Coupling}->{ACMIC} = "undef";        #Ancestor class class-method interaction import coupling
#  $hashMetrics->{Coupling}->{OCMIC} = "undef";        #Others class-method interaction import coupling
#  $hashMetrics->{Coupling}->{FCMEC} = "undef";        #Friends class-method interaction export coupling
#  $hashMetrics->{Coupling}->{DCMEC} = "undef";        #Descendents class-method interaction export coupling
#  $hashMetrics->{Coupling}->{OCMEC} = "undef";        #Others class-method interaction export coupling
#  $hashMetrics->{Coupling}->{OMMIC} = "undef";        #Others method-method interaction import coupling
#  $hashMetrics->{Coupling}->{IFMMIC} = "undef";       #Inverse friends method-method interaction import coupling
#  $hashMetrics->{Coupling}->{AMMIC} = "undef";        #Ancestor class method-method interaction import coupling
#  $hashMetrics->{Coupling}->{OMMEC} = "undef";        #Others method-method interaction export coupling
#  $hashMetrics->{Coupling}->{FMMEC} = "undef";        #Friends method-method interaction export coupling
#  $hashMetrics->{Coupling}->{DMMEC} = "undef";        #Descendents method-method interaction export coupling
#  $hashMetrics->{Coupling}->{CBI} = "undef";          #Degree of coupling of inheritance. From: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity metrics. COMPSAC 1996: 104-109.  
#  $hashMetrics->{Coupling}->{UCL} = "undef";          #Number of classes used in a class except for ancestors and children. From: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity metrics. COMPSAC 1996: 104-109.  
  # $hashMetrics->{Coupling}->{MPCNew} = "undef";       #Number of send statements in a class. From: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity metrics. COMPSAC 1996: 104-109.  
#  $hashMetrics->{Coupling}->{CC} = "undef";           #Class Coupling. From: C. Rajaraman, M.R. Lyu. Reliability and maintainability related software coupling metrics in C++ programs.
#  $hashMetrics->{Coupling}->{AMC} = "undef";          #Average Method Coupling. From: C. Rajaraman, M.R. Lyu. Reliability and maintainability related software coupling metrics in C++ programs.


  #inheritance metrics
  $hashMetrics->{Inheritance}->{NOC} = "undef";       #Number Of Child Classes
  $hashMetrics->{Inheritance}->{NOP} = "undef";       #Number Of Parent Classes
  $hashMetrics->{Inheritance}->{DIT} = "undef";       #Depth of Inheritance Tree
  $hashMetrics->{Inheritance}->{AID} = "undef";       #Average Inheritance Depth of a class(L.C.Briand, et al. Exloring the relationships between design measures and software quality in OO systems. JSS, vol. 51, 2000: 245-273.
  $hashMetrics->{Inheritance}->{CLD} = "undef";       #Class-to-Leaf Depth
  $hashMetrics->{Inheritance}->{NOD} = "undef";       #Number Of Descendents
  $hashMetrics->{Inheritance}->{NOA} = "undef";       #Number Of Ancestors
  $hashMetrics->{Inheritance}->{NMO} = "undef";       #Number of Methods Overridden
  $hashMetrics->{Inheritance}->{NMI} = "undef";       #Number of Methods Inherited
  $hashMetrics->{Inheritance}->{NMA} = "undef";       #Number Of Methods Added
  $hashMetrics->{Inheritance}->{SIX} = "undef";       #Specialization IndeX   =  NMO * DIT / (NMO + NMA + NMI)
  $hashMetrics->{Inheritance}->{PII} = "undef";       #Pure Inheritance Index. From: B.K. Miller, P. Hsia, C. Kung. Object-oriented architecture measures. 32rd Hawaii International Conference on System Sciences 1999    
  $hashMetrics->{Inheritance}->{SPA} = "undef";       #static polymorphism in ancestors
  $hashMetrics->{Inheritance}->{SPD} = "undef";       #static polymorphism in decendants
  $hashMetrics->{Inheritance}->{DPA} = "undef";       #dynamic polymorphism in ancestors
  $hashMetrics->{Inheritance}->{DPD} = "undef";       #dynamic polymorphism in decendants
  $hashMetrics->{Inheritance}->{SP} = "undef";        #static polymorphism in inheritance relations
  $hashMetrics->{Inheritance}->{DP} = "undef";        #dynamic polymorphism in inheritance relations
#  $hashMetrics->{Inheritance}->{CHM} = "undef";       #Class hierarchy metric. From J.Y. Chen, J.F. Lu. A new metric for OO design. IST, 35(4): 1993.
#  $hashMetrics->{Inheritance}->{DOR} = "undef";       #Degree of reuse by inheritance. From: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity metrics. COMPSAC 1996: 104-109.

  
  #size metrics 
  $hashMetrics->{Size}->{NMIMP} = "undef";     #Number Of Methods Implemented in a class
  $hashMetrics->{Size}->{NAIMP} = "undef";     #Number Of Attributes Implemented in a class
  $hashMetrics->{Size}->{loc}  = "undef";     #source lines of code
#  $hashMetrics->{Size}->{SLOCExe} = "undef";   #source lines of executable code
  $hashMetrics->{Size}->{stms}  = "undef";     #number of statements
#  $hashMetrics->{Size}->{stmsExe} = "undef";   #number of executable statements
  $hashMetrics->{Size}->{NM} = "undef";        #number of all methods (inherited, overriding, and non-inherited) methods of a class
  $hashMetrics->{Size}->{NA} = "undef";        #number of attributes in a class, both inherited and non-inherited
  $hashMetrics->{Size}->{Nmpub} = "undef";     #number of public methods implemented in a class
  $hashMetrics->{Size}->{NMNpub} = "undef";    #number of non-public methods implemented in a class
  $hashMetrics->{Size}->{NumPara} = "undef";   #sum of the number of parameters of the methods implemented in a class
  $hashMetrics->{Size}->{NIM} = "undef";       #Number of Instance Methods
  $hashMetrics->{Size}->{NCM} = "undef";       #Number of Class Methods
  $hashMetrics->{Size}->{NLM} = "undef";       #Number of Local Methods
  $hashMetrics->{Size}->{AvgSLOC} = "undef";   #Average Source Lines of Code
#  $hashMetrics->{Size}->{AvgSLOCExe} = "undef";#Average Source Lines of Executable Code

  #cohesion metrics
  # $hashMetrics->{Cohesion}->{LCOM1} = "undef";
  # $hashMetrics->{Cohesion}->{LCOM2} = "undef";
  # $hashMetrics->{Cohesion}->{LCOM3} = "undef";
  # $hashMetrics->{Cohesion}->{LCOM4} = "undef";
  # $hashMetrics->{Cohesion}->{Co}    = "undef";
#  $hashMetrics->{Cohesion}->{NewCo} = "undef";
  # $hashMetrics->{Cohesion}->{LCOM5} = "undef";
#  $hashMetrics->{Cohesion}->{NewLCOM5} = "undef";  #also called NewCoh/Coh
  # $hashMetrics->{Cohesion}->{LCOM6} = "undef";     #based on parameter names. From: J.Y. Chen, J.F. Lu. A new metric for OO design. IST, 35(4): 1993.
  # $hashMetrics->{Cohesion}->{LCC}   = "undef";     #Loose Class Cohesion
  # $hashMetrics->{Cohesion}->{TCC}   = "undef";     #Tight Class Cohesion   
  # $hashMetrics->{Cohesion}->{ICH}   = "undef";     #Information-flow-based Cohesion
  # $hashMetrics->{Cohesion}->{DCd}   = "undef";     #Degree of Cohesion based Direct relations between the public methods 
  # $hashMetrics->{Cohesion}->{DCi}   = "undef";     #Degree of Cohesion based Indirect relations between the public methods
#  $hashMetrics->{Cohesion}->{CBMC}  = "undef";   
#  $hashMetrics->{Cohesion}->{ICBMC} = "undef"; 
#  $hashMetrics->{Cohesion}->{ACBMC} = "undef"; 
#  $hashMetrics->{Cohesion}->{C3}    = "undef"; 
#  $hashMetrics->{Cohesion}->{LCSM}  = "undef";     #Lack of Conceptual similarity between Methods
  # $hashMetrics->{Cohesion}->{OCC}   = "undef";     #Opitimistic Class Cohesion
  # $hashMetrics->{Cohesion}->{PCC}   = "undef";     #Pessimistic Class Cohesion
  # $hashMetrics->{Cohesion}->{CAMC}  = "undef";     #Cohesion Among Methods in a Class
#  $hashMetrics->{Cohesion}->{iCAMC} = "undef";     #������������ֵ���͵�CAMC
#  $hashMetrics->{Cohesion}->{CAMCs} = "undef";     #����self���͵�CAMC
#  $hashMetrics->{Cohesion}->{iCAMCs}= "undef";     #������������ֵ���ͺ�self���͵�CAMC
  # $hashMetrics->{Cohesion}->{NHD}   = "undef";     #Normalized Hamming Distance metric
#  $hashMetrics->{Cohesion}->{iNHD}  = "undef";  
#  $hashMetrics->{Cohesion}->{NHDs}  = "undef"; 
#  $hashMetrics->{Cohesion}->{iNHDs} = "undef";     
  # $hashMetrics->{Cohesion}->{SNHD}  = "undef";     
#  $hashMetrics->{Cohesion}->{iSNHD}  = "undef";     
#  $hashMetrics->{Cohesion}->{SNHDs}  = "undef";    
#  $hashMetrics->{Cohesion}->{iSNHDs}  = "undef";    
  # $hashMetrics->{Cohesion}->{SCOM}  = "undef";     #Sensitive Class Cohesion Metric. From: International Journal of Information Theories & Applications. Vol. 13, No. 1, 2006: 82-91 
  # $hashMetrics->{Cohesion}->{CAC}  = "undef";      #Class Abstraction Cohesion. From: B.K. Miller, P. Hsia, C. Kung. Object-oriented architecture measures. 32rd Hawaii International Conference on System Sciences 1999

  #Other metrics
  # $hashMetrics->{Other}->{OVO} = "undef";          #parametric overloading metric 
  # $hashMetrics->{Other}->{MI} = "undef";        #Maintainability Index

 
  return $hashMetrics;
} # End sub defineMetrics



sub computeMetrics{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	my $sBriandCouplingMetrics = shift;	
	my $hashMetrics = shift;

  #-----------------����һЩ���õ�����, ����߼���Ч��--------------------------
  	my %ancestorHash = ();
  	my $ancestorLevel;
  	$ancestorLevel = getAncestorClasses($sClass, \%ancestorHash); 	
  
  	my %descendentClassHash;
  	getDescendentClasses($sClass, \%descendentClassHash);



	# $hashMetrics->{Size}->{NMIMP} = NMIMP($sClass); 
	# $hashMetrics->{Size}->{NAIMP} = NAIMP($sClass); 

   
  #----------------compute complexity metrics------------------------
  
  	# my $start = getTimeInSecond();
  	$hashMetrics->{Complexity}->{CDE} = CDE($sClass);
  	# reportComputeTime($start, "CDE");  
  
  	# $start = getTimeInSecond();
  	$hashMetrics->{Complexity}->{CIE} = CIE($sClass);
  	# reportComputeTime($start, "CIE");  
  
  	$hashMetrics->{Complexity}->{WMC} = WMC($sClass);
 	  $hashMetrics->{Complexity}->{SDMC} = SDMC($sClass);
  	$hashMetrics->{Complexity}->{AvgWMC} = AvgWMC($sClass);
  	$hashMetrics->{Complexity}->{CCMax} = CCMax($sClass);
  	$hashMetrics->{Complexity}->{NTM} = NTM($sClass);
  
#  	$start = getTimeInSecond();
#  	my ($valueCC1, $valueCC2, $valueCC3) = CComplexitySerires($sClass, \%ancestorHash, $ancestorLevel); 
#  	$hashMetrics->{Complexity}->{CC1} = $valueCC1;
#  	$hashMetrics->{Complexity}->{CC2} = $valueCC2;
#  	$hashMetrics->{Complexity}->{CC3} = $valueCC3;
#  	reportComputeTime($start, "CC1, CC2, CC3");  
  

    
  #----------------compute coupling metrics------------------------  
	$hashMetrics->{Coupling}->{CBO} = $sClass->metric("CountClassCoupled");
	
	# $start = getTimeInSecond();
	# $hashMetrics->{Coupling}->{RFC} = RFC($sClass, \%ancestorHash, $ancestorLevel);
#	$hashMetrics->{Coupling}->{RFC1} = RFC1($sClass, \%ancestorHash, $ancestorLevel);
	# reportComputeTime($start, "RFC");  
	
	# $start = getTimeInSecond();
	# ($hashMetrics->{Coupling}->{MPC}, $hashMetrics->{Coupling}->{MPCNew}) = MPCSeries($sClass, $sAllClassNameHash);
	# reportComputeTime($start, "MPC");  
	
	# $start = getTimeInSecond();
	($hashMetrics->{Coupling}->{DAC}, $hashMetrics->{Coupling}->{DACquote}) = DAC($sClass, $sAllClassNameHash);
	# reportComputeTime($start, "DAC");  
	
	# $start = getTimeInSecond();
	my $valueICP = ICP($sClass, $sAllClassNameHash);
	$hashMetrics->{Coupling}->{ICP} = $valueICP;	
	my $valueIHICP = IHICP($sClass, \%ancestorHash);
	$hashMetrics->{Coupling}->{IHICP} = $valueIHICP;
	# reportComputeTime($start, "ICP");  
	
	$hashMetrics->{Coupling}->{NIHICP} = NIHICP($valueICP, $valueIHICP);	
	
	my $classKey = getClassKey($sClass);
	
#	$hashMetrics->{Coupling}->{IFCAIC} = $sBriandCouplingMetrics->{$classKey}->{IFCAIC};
#	$hashMetrics->{Coupling}->{ACAIC} = $sBriandCouplingMetrics->{$classKey}->{ACAIC};
#	$hashMetrics->{Coupling}->{OCAIC} = $sBriandCouplingMetrics->{$classKey}->{OCAIC};
#	$hashMetrics->{Coupling}->{FCAEC} = $sBriandCouplingMetrics->{$classKey}->{FCAEC};
#	$hashMetrics->{Coupling}->{DCAEC} = $sBriandCouplingMetrics->{$classKey}->{DCAEC};
#	$hashMetrics->{Coupling}->{OCAEC} = $sBriandCouplingMetrics->{$classKey}->{OCAEC};
#	$hashMetrics->{Coupling}->{IFCMIC} = $sBriandCouplingMetrics->{$classKey}->{IFCMIC};
#	$hashMetrics->{Coupling}->{ACMIC} = $sBriandCouplingMetrics->{$classKey}->{ACMIC};
#	$hashMetrics->{Coupling}->{OCMIC} = $sBriandCouplingMetrics->{$classKey}->{OCMIC};
#	$hashMetrics->{Coupling}->{FCMEC} = $sBriandCouplingMetrics->{$classKey}->{FCMEC};
#	$hashMetrics->{Coupling}->{DCMEC} = $sBriandCouplingMetrics->{$classKey}->{DCMEC};
#	$hashMetrics->{Coupling}->{OCMEC} = $sBriandCouplingMetrics->{$classKey}->{OCMEC};
#	$hashMetrics->{Coupling}->{IFMMIC} = $sBriandCouplingMetrics->{$classKey}->{IFMMIC};
#	$hashMetrics->{Coupling}->{AMMIC} = $sBriandCouplingMetrics->{$classKey}->{AMMIC};
#	$hashMetrics->{Coupling}->{OMMIC} = $sBriandCouplingMetrics->{$classKey}->{OMMIC};
#	$hashMetrics->{Coupling}->{FMMEC} = $sBriandCouplingMetrics->{$classKey}->{FMMEC};
#	$hashMetrics->{Coupling}->{DMMEC} = $sBriandCouplingMetrics->{$classKey}->{DMMEC};
#	$hashMetrics->{Coupling}->{OMMEC} = $sBriandCouplingMetrics->{$classKey}->{OMMEC};		
	
#	$start = getTimeInSecond();
#	$hashMetrics->{Coupling}->{CBI} = CBI($sClass, \%descendentClassHash);
#	reportComputeTime($start, "CBI");  
#	
#	$start = getTimeInSecond();
#	$hashMetrics->{Coupling}->{UCL} = UCL($sClass, $sAllClassNameHash, \%ancestorHash, \%descendentClassHash);
#	reportComputeTime($start, "UCL");  
#	
#	$start = getTimeInSecond();
#	($hashMetrics->{Coupling}->{CC}, $hashMetrics->{Coupling}->{AMC}) = CCAndAMC($sClass);
#	reportComputeTime($start, "CCandAMC");  
	
	#----------------compute inheritance metrics------------------------  
	# $start = getTimeInSecond();	
	InheritanceSeries($sClass, $sAllClassNameHash, \%ancestorHash, $ancestorLevel, \%descendentClassHash, $hashMetrics);
	# reportComputeTime($start, "InheritanceSeries");  
	

#	$start = getTimeInSecond();
#  	my $trtr = $sBriandCouplingMetrics->{totalNOD};  
#	my $valueDOR = DOR($sClass, $sAllClassNameHash, \%descendentClassHash, $trtr);
#	$hashMetrics->{Inheritance}->{DOR} = $valueDOR;
#	reportComputeTime($start, "DOR"); 
#	
		
	#----------------compute size metrics------------------------  
	$hashMetrics->{Size}->{NMIMP} = NMIMP($sClass); 
	$hashMetrics->{Size}->{NAIMP} = NAIMP($sClass); 
	$hashMetrics->{Size}->{loc} = UnderstandSLOC($sClass);
#	$hashMetrics->{Size}->{SLOCExe} = SLOCExe($sClass);
	$hashMetrics->{Size}->{stms} = $sClass->metric("CountStmt");
#	$hashMetrics->{Size}->{stmsExe} = $sClass->metric("CountStmtExe");
  $hashMetrics->{Size}->{NM} = NM($sClass, \%ancestorHash);
  $hashMetrics->{Size}->{NA} = NA($sClass, \%ancestorHash);
	$hashMetrics->{Size}->{Nmpub} = Nmpub($sClass);
	$hashMetrics->{Size}->{NMNpub} = NMNpub($sClass);
	$hashMetrics->{Size}->{NumPara} = NumPara($sClass);
	$hashMetrics->{Size}->{NIM} = NIM($sClass);
	$hashMetrics->{Size}->{NCM} = NCM($sClass);
	$hashMetrics->{Size}->{NLM} = NLM($sClass);
	$hashMetrics->{Size}->{AvgSLOC} = AvgSLOCPerMethod($sClass);
#	$hashMetrics->{Size}->{AvgSLOCExe} = AvgSLOCExePerMethod($sClass);

	
	#----------------compute cohesion metrics------------------------  
	# $start = getTimeInSecond();
	# LCOMSeriesAndCAC($sClass, $hashMetrics); #ע��, Ϊ��Ч��CACҲ�����������
	# TCCLCCSeries($sClass, $hashMetrics);
	# $hashMetrics->{Cohesion}->{ICH} = ICH($sClass);
	#CBMCSeries($sClass, $hashMetrics);
	# OCCAndPCC($sClass, $hashMetrics);
	# CAMCSeries($sClass, $hashMetrics);
	# reportComputeTime($start, "cohesion");
	
	# $start = getTimeInSecond();
	# $hashMetrics->{Cohesion}->{SCOM} = SCOM($sClass);
	# reportComputeTime($start, "SCOM");
	
#	$start = getTimeInSecond();
#	($hashMetrics->{Cohesion}->{C3}, $hashMetrics->{Cohesion}->{LCSM}) = C3($sClass);   
#	reportComputeTime($start, "C3");  
	
	
	#----------------compute other metrics------------------------  
	# $hashMetrics->{Other}->{OVO} = OVO($sClass);
	# $hashMetrics->{Other}->{MI} = MI($sClass);
	
	return $hashMetrics;	
} # End sub computeMetrics



sub RFC{
	my $sClass = shift;
	my $sAncestorHash = shift;
	my $sAncestorLevel = shift;	
	
	my %responseSet;	

  print "\t\t\t computing RFC..." if ($debug);
	
	my %totalMethods = (); #������е����з���: �̳з�overriding�� + overriding + �����ӵ�
	
	#�õ�ǰ��ķ��������Խ��г�ʼ��
	my @methodArray = getEntsInClass($sClass, "define", "function ~unknown, method ~unknown");
	foreach my $func (@methodArray){
		my $signature = getFuncSignature($func, 1);
		$totalMethods{$signature} = $func;
		my $key = getLastName($sClass->name())."::".getFuncSignature($func, 1);				
		$responseSet{$key} = 1;		
	}
		
	#����̳еķ���
	foreach my $level (sort keys %{$sAncestorLevel}){
		my %ancestorHash = %{$sAncestorLevel->{$level}};
		
		foreach my $classKey (keys %ancestorHash){
			my $ancestorClass = $sAncestorHash->{$classKey};
			
			#----��Ӽ̳з�overiding�ķ���-----------
			my @ancestorMethodArray = getEntsInClass($ancestorClass, "define", "function ~private ~unknown, method ~private ~unknown");
			foreach my $func (@ancestorMethodArray){
				my $signature = getFuncSignature($func, 1);
				next if (exists $totalMethods{$signature}); #��������overriding��, ��������
				$totalMethods{$signature} = $func;
				my $key = getLastName($ancestorClass->name())."::".getFuncSignature($func, 1);				
		    $responseSet{$key} = 1;
			}			
		}		
	}	
	
	

	my @allMethodArray = (values %totalMethods);
		
	
	while (@allMethodArray > 0){
		my $currentFunc = shift @allMethodArray;
		
		my %tempRS = ();
		PIM($currentFunc, \%tempRS);
		
		foreach my $currentKey (keys %tempRS){			
			next if (exists $responseSet{$currentKey});		#��һ�����Ҫ, �����п��������ѭ��	
			
			$responseSet{$currentKey} = 1;
			push @allMethodArray, $tempRS{$currentKey}->{funcEnt};
		}
	}
	
	my $result = 0;	
	$result = (keys %responseSet);

  
#	foreach my $key (sort keys %responseSet){
#		print "\t\t", $key, "\n";
#	}
#	print "---------------------\n";
	
	print ".....RFC END\n" if ($debug);
	return $result;		
}#END sub RFC


sub RFC1{
	my $sClass = shift;
	my $sAncestorHash = shift;
	my $sAncestorLevel = shift;	

	my %responseSet;	

  print "\t\t\t computing RFC1..." if ($debug);
	
	my %totalMethods = (); #������е����з���: �̳з�overriding�� + overriding + �����ӵ�
	
	#�õ�ǰ��ķ��������Խ��г�ʼ��
	my @methodArray = getEntsInClass($sClass, "define", "function ~unknown, method ~unknown");
	foreach my $func (@methodArray){
		my $signature = getFuncSignature($func, 1);
		$totalMethods{$signature} = $func;
		my $key = getLastName($sClass->name())."::".getFuncSignature($func, 1);				
		$responseSet{$key} = 1;		
	}
		
	#����̳еķ���
	foreach my $level (sort keys %{$sAncestorLevel}){
		my %ancestorHash = %{$sAncestorLevel->{$level}};
		
		foreach my $classKey (keys %ancestorHash){
			my $ancestorClass = $sAncestorHash->{$classKey};
			
			#----��Ӽ̳з�overiding�ķ���-----------
			my @ancestorMethodArray = getEntsInClass($ancestorClass, "define", "function ~private ~unknown, method ~private ~unknown");
			foreach my $func (@ancestorMethodArray){
				my $signature = getFuncSignature($func, 1);
				next if (exists $totalMethods{$signature}); #��������overriding��, ��������
				$totalMethods{$signature} = $func;
				my $key = getLastName($ancestorClass->name())."::".getFuncSignature($func, 1);				
		    $responseSet{$key} = 1;
			}			
		}		
	}	
	
	

	my @allMethodArray = (values %totalMethods);
	
	foreach my $func (@allMethodArray){
		my %tempRS=();
		PIM($func, \%tempRS);
		
		foreach my $key (keys %tempRS){			
			$responseSet{$key} = 1;
		}
	}
		
	my $result = 0;
	
	$result = (keys %responseSet);
	
#	foreach my $key (sort keys %responseSet){
#		print "\t\t", $key, "\n";
#	}
#	print "---------------------\n";
	
	print "...RFC1 END\n" if ($debug);
	return $result;	
}#END sub RFC1



sub MPCSeries{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	
	print "\t\t\t computing MPCSeries..." if ($debug);

	my @methodArray = getRefsInClass($sClass, "define", "function ~unresolved ~unknown, method ~unresolved ~unknown");    
	
	my %calledMethodHash; #��¼����ǰ�������õ��������ж���ķ���. key�Ƿ�����, value��{Ent=>  , count => }
	                      #����, Ent�Ƿ���ʵ��, count�Ǳ����õĴ���
  
  my $callingClassKey = getClassKey($sClass);
  
  my $valueMPC = 0;
  
  
  foreach my $method (@methodArray){
  	my @calledFuncSet = $method->ent()->refs("call", "function, method");
  	foreach my $func (@calledFuncSet){
  		my $calledClass = $func->ent()->ref("Definein", "Class");
  		next if (!$calledClass);  		  		
  		next if ($calledClass->ent()->library() =~ m/Standard/i);
  		
  		my $calledClassKey = getClassKey($calledClass->ent());
  		#next if (!exists $sAllClassNameHash->{$calledClassKey});  		
  		next if ($callingClassKey eq $calledClassKey); 
  		
  		$valueMPC++;  		
  		
  		my $methodSignature = getFuncSignature($func->ent(), 1);
  		my $key = $calledClassKey.$methodSignature;
  		
  		$calledMethodHash{$key}->{Ent} = $func->ent();
		
  		if (!exists $calledMethodHash{$key}){
  			$calledMethodHash{$key}->{Count} = 1;
  		}
  		else{
  			$calledMethodHash{$key}->{Count}++;
  		}  		
  	}
  }
  
  my $valueMPCNew = 0;
  
  foreach my $key (keys %calledMethodHash){
  	my $func = $calledMethodHash{$key}->{Ent};
  	my $count = $calledMethodHash{$key}->{Count};
  	
  	my $stmtNo = $func->metric("CountStmt");
  	
  	my $reserveWordsNo = getNoReserveWords($func);
  	
#  	print "\t\t calledFunc = ", $func->name(), "\n";
#  	print "\t\t count = ", $count, "\n";  	
#  	print "\t\t reserveWordsNo = ", $reserveWordsNo, "\n";  	
#  	print "\t\t stmtNo = ", $stmtNo, "\n";  
  	
  	$valueMPCNew = $valueMPCNew +  ($stmtNo + $reserveWordsNo) * $count;
  }
  
  
  print "...MPCSeries END\n" if ($debug);
  
  return wantarray? ($valueMPC, $valueMPCNew): $valueMPC;		
}#END sub MPCnew



sub DAC{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	
#	print "\t\t\t computing DAC...\n";
	
	my @attributeArray = getRefsInClass($sClass, "define","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");
	
	my $valueDAC = 0;
	my $valueDACquote = 0;
	my %dacClassHash;
  
	foreach my $attribute (@attributeArray){
		my $attributeClass = $attribute->ent()->ref("Typed", "Class");
		
		next if (!$attributeClass);
		
#		print "\t\t attribute = ", $attribute->ent()->name(), "\n";
#		print "\t\t attribute Class = ", $attributeClass->ent()->name(), "\n";
		
		next if ($attributeClass->ent()->library() =~ m/Standard/i);
		
#		print "\t\t attribute = ", $attribute->ent()->name(), "\n";
#		print "\t\t classType = ", $attributeClass->ent()->name(), "\n";

		my $attributeClassKey = getClassKey($attributeClass->ent());		
		$dacClassHash{$attributeClassKey} = 1;
		#next if (!exists $sAllClassNameHash->{$attributeClassKey});		
		   
		$valueDAC++;   
	}
	
	$valueDACquote = scalar (keys %dacClassHash);
	
	return wantarray?($valueDAC, $valueDACquote): $valueDAC;
}#END sub DAC


sub ICP{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	
	my @methodArray = getEntsInClass($sClass, "define","function  ~unresolved ~unknown, method  ~unresolved ~unknown");    
	
	my $result = 0;
  
  my $callingClassKey = getClassKey($sClass);

  #ֻ���Ƕ�̬�ĵ���    
  foreach my $method (@methodArray){
  	my %polyCalledFuncSet;
  	PIM($method, \%polyCalledFuncSet);
  	
  	foreach my $key (sort keys %polyCalledFuncSet){
  		my $calledFuncEnt = $polyCalledFuncSet{$key}->{funcEnt};
  		my $callCount = $polyCalledFuncSet{$key}->{callCount};
  		
  		my $calledClass = $calledFuncEnt->ref("Definein", "Class");
  		next if (!$calledClass);		
  		next if ($calledClass->ent()->library() =~ m/Standard/i);
  		
  		my $calledClassKey = getClassKey($calledClass->ent());
  		#next if (!exists $sAllClassNameHash->{$calledClassKey});
  		
  		next if ($callingClassKey eq $calledClassKey);
  		
  		my @parameterSet = $calledFuncEnt->ents("Define", "Parameter");
  		
  		$result = $result + (@parameterSet + 1) * $callCount;  
  	}
  }
  
#  #ֻ���Ǿ�̬�ĵ���  
#  foreach my $method (@methodArray){
#  	my @calledFuncSet = $method->refs("call", "function  ~unresolved ~unknown, method  ~unresolved ~unknown");
#  	foreach my $func (@calledFuncSet){
#  		my $calledClass = $func->ent()->ref("Definein", "Class ~unknown ~unresovled");
#  		next if (!$calledClass);
#  		
#  		my $calledClassKey = getClassKey($calledClass->ent());
#  		
#     next if (!exists $sAllClassNameHash->{$calledClassKey});
#  		next if ($callingClassKey eq $calledClassKey);
#  		  		 
#  		my @parameterSet = $func->ent()->ents("Define", "Parameter");
#  		$result = $result + @parameterSet + 1;  		
#  	}
#  }
#  

  return $result;	
}#END sub ICP



sub IHICP{
	my $sClass = shift;
	my $sAncestorHash = shift;
	
	#����������ļ���
	my %ancestorHash = %{$sAncestorHash};	#֮������Hash��, ���Ƕ�̳е����
	
	#����IHICP
	my @methodArray = getEntsInClass($sClass, "define","function  ~unresolved ~unknown, method  ~unresolved ~unknown");    
	
	my $result = 0; 

  #ֻ���Ƕ�̬�ĵ���    
  foreach my $method (@methodArray){
  	my %polyCalledFuncSet;
  	PIM($method, \%polyCalledFuncSet);
  	
  	foreach my $key (sort keys %polyCalledFuncSet){
  		my $calledFuncEnt = $polyCalledFuncSet{$key}->{funcEnt};
  		my $callCount = $polyCalledFuncSet{$key}->{callCount};
  		
  		my $calledClass = $calledFuncEnt->ref("Definein", "Class");
  		next if (!$calledClass);
  		next if ($calledClass->ent()->library() =~ m/Standard/i);
  		
  		my $calledClassKey = getClassKey($calledClass->ent());
  		next if (!exists $ancestorHash{$calledClassKey});
  		
  		my @parameterSet = $calledFuncEnt->ents("Define", "Parameter");
  		
  		$result = $result + (@parameterSet + 1) * $callCount;  
  	}
  }


#  #ֻ���Ǿ�̬�ĵ��� 
#  foreach my $method (@methodArray){
#  	my @calledFuncSet = $method->refs("call", "function  ~unresolved ~unknown, method  ~unresolved ~unknown");
#  	foreach my $func (@calledFuncSet){
#  		my $calledClass = $func->ent()->ref("Definein", "Class ~unknown ~unresovled");
#  		next if (!$calledClass);
#  		next if ($calledClass->ent()->library() =~ m/Standard/i);
#  		
#  		my $calledClassKey = getClassKey($calledClass->ent());
#  		
#  		next if (!exists $ancestorHash{$calledClassKey});
#  		
#  		my @parameterSet = $func->ent()->ents("Define", "Parameter");  		
#  		$result = $result + @parameterSet + 1;  		
#  	}
#  }
  
  return $result;		
}#END sub IHICP


sub NIHICP{
  my $sICP = shift;
	my $sIHICP = shift;	
  
	my $result = $sICP - $sIHICP;	
	
	return $result;		
}#END sub NIHICP




sub getBriandCouplingMetrics{	
	my $sAllClassNameHash = shift;
	
	my $BriandCouplingMetrics = {}; #��Ž��:  ->{����}->{������} = value;
	
	my $CAInteractionMatrix = {}; #������-���Խ�������
	my $CMInteractionMatrix = {}; #������-������������
	my $MMInteractionMatrix = {}; #���ķ���-������������
	
	$CAInteractionMatrix->{myTotalSum} = 0; #�����ܺ�, �Ա����һ������other��������, such as OCAEC
	$CMInteractionMatrix->{myTotalSum} = 0;
	$MMInteractionMatrix->{myTotalSum} = 0;
	
	print "computing Briand Coupling metrics, please wait....\n";
	my $count = 0;
	
	foreach my $currentClassKey (keys %{$sAllClassNameHash}){
		my $currentClass = $sAllClassNameHash->{$currentClassKey};				
		
		print "Analyzing class ", $count, "...\n";
		$count++;
		
    #ɨ����, �����-���Ծ���		
	  my @attributeArray = getRefsInClass($currentClass, "define","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");	
	  my $result = 0;
  
	  foreach my $attribute (@attributeArray){
		  my $attributeClass = $attribute->ent()->ref("Typed", "Class");
		  next if (!$attributeClass);		
		  next if ($attributeClass->ent()->library() =~ m/Standard/i);

      my $attributeClassKey = getClassKey($attributeClass->ent());
   		
   		#next if (!exists $sAllClassNameHash->{$attributeClassKey}); #�������Ӧ����, ������   		
   		next if ($attributeClassKey eq $currentClassKey); #��Ϊ�����, ����ֻ���ǵ�ǰ����������֮��Ĺ�ϵ

   		$CAInteractionMatrix->{Matrix}->{$currentClassKey}->{$attributeClassKey}++;   		
   		$CAInteractionMatrix->{myTotalSum}++;		
   	}
   	
   	
   	#++++++++++++++++++ɨ����, �����-��������++++++++++++++++++++++		
   	my %addedMethodHash; 
	  getAddedMethods($currentClass, \%addedMethodHash);
	
	  foreach my $key (keys %addedMethodHash){
		  my $func = $addedMethodHash{$key};		
		  my @parameters = $func->ents("Define", "Parameter");		
		  #����������ÿ������������
		  foreach my $para (@parameters){			
			  my $parameterClass = $para->ref("Typed", "Class");			
			  next if (!$parameterClass);			  			
			  next if ($parameterClass->ent()->library() =~ m/Standard/i);

        my $parameterClassKey = getClassKey($parameterClass->ent());
			  
			  #next if (!exists $sAllClassNameHash->{$parameterClassKey}); #�������Ӧ����, ������
			  next if ($parameterClassKey eq $currentClassKey); #��Ϊ�����, ����ֻ���ǵ�ǰ����������֮��Ĺ�ϵ
			  
			  $CMInteractionMatrix->{Matrix}->{$currentClassKey}->{$parameterClassKey}++;
   		  $CMInteractionMatrix->{myTotalSum}++;				  
		  }
		  
		
		  #���������ķ�������
		  my $returnClass = $func->ref("Typed", "Class");			
			next if (!$returnClass);			  					  
			next if ($returnClass->ent()->library() =~ m/Standard/i);
    
		  my $returnClassKey = getClassKey($returnClass->ent());
		  
			#next if (!exists $sAllClassNameHash->{$returnClassKey}); #�������Ӧ����, ������
			next if ($returnClassKey eq $currentClassKey); #��Ϊ�����, ����ֻ���ǵ�ǰ����������֮��Ĺ�ϵ		  
			
			$CMInteractionMatrix->{Matrix}->{$currentClassKey}->{$returnClassKey}++;
   		$CMInteractionMatrix->{myTotalSum}++;
	  }
   	
   	
   	#++++++++++++++++++ɨ����, ��䷽��-��������++++++++++++++++++++++		
  	my @methodArray = getRefsInClass($currentClass, "define", "function ~unresolved ~unknown, method ~unresolved ~unknown");    	
	
    #ͳ���Ե�ǰ���е����������з����ķ�����Ŀ
    foreach my $method (@methodArray){
  		my @calledFuncSet = $method->ent()->refs("call", "function ~unresolved ~unknown, method ~unresolved ~unknown");
  	  foreach my $func (@calledFuncSet){
  		  my $calledClass = $func->ent()->ref("Definein", "Class");
  			next if (!$calledClass);  		  		
  			next if ($calledClass->ent()->library() =~ m/Standard/i);
  			
  			my $calledClassKey = getClassKey($calledClass->ent()); 
  		
  			#next if (!exists $sAllClassNameHash->{$calledClassKey}); #�������Ӧ����, ������
  			next if ($calledClassKey eq $currentClassKey); #��Ϊ�����, ����ֻ���ǵ�ǰ����������֮��Ĺ�ϵ	
  			
  			$MMInteractionMatrix->{Matrix}->{$currentClassKey}->{$calledClassKey}++;
   			$MMInteractionMatrix->{myTotalSum}++;
  		}
  		
  		#ͳ����$sClassD�з���Ϊ������(��$sClassC)������Ŀ
  		#������....
 	  }  
  }

  #---------------------����18������Զ���--------------------		  
	BriandCouplingSeries($sAllClassNameHash, $CAInteractionMatrix, $CMInteractionMatrix, $MMInteractionMatrix, 
	                     $BriandCouplingMetrics);
	
	return $BriandCouplingMetrics;
}#END sub getBriandCouplingMetrics



sub BriandCouplingSeries{
	#����һ�����, �Ա��ʡ����������, ������, ��Ԫ��, ������Ԫ��, �Լ��������ʱ��
	my $sAllClassNameHash = shift;
	my $sClassAttributeMatrix = shift;
	my $sClassMethodMatrix = shift;
	my $sMethodMethodMatrix = shift;
	my $sBriandCouplingMetrics = shift; 
	
	
#	print "CA matrix: \n";
#	
#	print "\total sum = ", $sClassAttributeMatrix->{myTotalSum}, "\n";
#	foreach my $source (keys %{$sClassAttributeMatrix->{Matrix}}){
#		my %tempHash = %{$sClassAttributeMatrix->{Matrix}->{$source}};
#		foreach my $dest (keys %tempHash){
#			print "\t\t(", $source, ",", $dest, ",", $sClassAttributeMatrix->{Matrix}->{$source}->{$dest}, ")\n";
#		}
#	}
#	
#
#	print "CM matrix: \n";
#	
#	print "\total sum = ", $sClassMethodMatrix->{myTotalSum}, "\n";
#	foreach my $source (keys %{$sClassMethodMatrix->{Matrix}}){
#		my %tempHash = %{$sClassMethodMatrix->{Matrix}->{$source}};
#		foreach my $dest (keys %tempHash){
#			print "\t\t(", $source, ",", $dest, ",", $sClassMethodMatrix->{Matrix}->{$source}->{$dest}, ")\n";
#		}
#	}
#	
#	
#	print "MM matrix: \n";
#	
#	print "\total sum = ", $sMethodMethodMatrix->{myTotalSum}, "\n";
#	foreach my $source (keys %{$sMethodMethodMatrix->{Matrix}}){
#		my %tempHash = %{$sMethodMethodMatrix->{Matrix}->{$source}};
#		foreach my $dest (keys %tempHash){
#			print "\t\t(", $source, ",", $dest, ",", $sMethodMethodMatrix->{Matrix}->{$source}->{$dest}, ")\n";
#		}
#	}	
	
	my $count = 0;
	
	foreach my $classKey (keys %{$sAllClassNameHash}){
		my $currentClass = $sAllClassNameHash->{$classKey};
		
		$count++;
		print "Computing metrics for class ", $count, "...\n";
		
    my %ancestorClassHash; 
    getAncestorClasses($currentClass, \%ancestorClassHash);	
    
    my %descendentClassHash; 
    getDescendentClasses($currentClass, \%descendentClassHash);  
    
    my %friendClassHash; 
    getFriendClasses($currentClass, \%friendClassHash);  

    my %inverseFriendClassHash; 
    getInverseFriendClasses($currentClass, \%inverseFriendClassHash);	
    
    my %otherClassHash;
    getOtherClasses($currentClass, $sAllClassNameHash, \%ancestorClassHash,\%descendentClassHash, \%friendClassHash, \%inverseFriendClassHash,\%otherClassHash);
    	                             
    my %OthersDiffFriendHash;
	  getDiffHash(\%otherClassHash, \%friendClassHash, \%OthersDiffFriendHash);		                             

		my %OthersDiffInverseFriendHash;
	  getDiffHash(\%otherClassHash, \%friendClassHash, \%OthersDiffInverseFriendHash);		  
	  
	
	  $sBriandCouplingMetrics->{$classKey}->{IFCAIC} = SumOfColumn($currentClass, \%inverseFriendClassHash, $sClassAttributeMatrix);		  
	  $sBriandCouplingMetrics->{$classKey}->{ACAIC} = SumOfColumn($currentClass, \%ancestorClassHash, $sClassAttributeMatrix);	
	  $sBriandCouplingMetrics->{$classKey}->{OCAIC} = SumOfColumn($currentClass, "All", $sClassAttributeMatrix)
	                                                  - SumOfColumn($currentClass, \%OthersDiffFriendHash, $sClassAttributeMatrix);
	                                                  	  
	  
    $sBriandCouplingMetrics->{$classKey}->{FCAEC} = SumOfRow(\%friendClassHash, $currentClass, $sClassAttributeMatrix);
	  $sBriandCouplingMetrics->{$classKey}->{DCAEC} = SumOfRow(\%descendentClassHash, $currentClass, $sClassAttributeMatrix);	
	  $sBriandCouplingMetrics->{$classKey}->{OCAEC} = SumOfRow("All", $currentClass, $sClassAttributeMatrix)
	                                                  - SumOfRow(\%OthersDiffInverseFriendHash, $currentClass, $sClassAttributeMatrix);	
	  
	
	  $sBriandCouplingMetrics->{$classKey}->{IFCMIC} = SumOfColumn($currentClass, \%inverseFriendClassHash, $sClassMethodMatrix);		  
	  $sBriandCouplingMetrics->{$classKey}->{ACMIC} = SumOfColumn($currentClass, \%ancestorClassHash, $sClassMethodMatrix);	
	  $sBriandCouplingMetrics->{$classKey}->{OCMIC} = SumOfColumn($currentClass, "All", $sClassMethodMatrix)
	                                                  - SumOfColumn($currentClass, \%OthersDiffFriendHash, $sClassMethodMatrix);	  

	
	 	$sBriandCouplingMetrics->{$classKey}->{FCMEC} = SumOfRow(\%friendClassHash, $currentClass, $sClassMethodMatrix);
	  $sBriandCouplingMetrics->{$classKey}->{DCMEC} = SumOfRow(\%descendentClassHash, $currentClass, $sClassMethodMatrix);
	  $sBriandCouplingMetrics->{$classKey}->{OCMEC} = SumOfRow("All", $currentClass, $sClassMethodMatrix)
	                                                  - SumOfRow(\%OthersDiffInverseFriendHash, $currentClass, $sClassMethodMatrix);
	
		
	  $sBriandCouplingMetrics->{$classKey}->{IFMMIC} = SumOfColumn($currentClass, \%inverseFriendClassHash, $sMethodMethodMatrix);		  
	  $sBriandCouplingMetrics->{$classKey}->{AMMIC} = SumOfColumn($currentClass, \%ancestorClassHash, $sMethodMethodMatrix);
	  $sBriandCouplingMetrics->{$classKey}->{OMMIC} = SumOfColumn($currentClass, "All", $sMethodMethodMatrix)
	                                                  - SumOfColumn($currentClass, \%OthersDiffFriendHash, $sMethodMethodMatrix);	  
	                                                  	  
	
	  $sBriandCouplingMetrics->{$classKey}->{FMMEC} = SumOfRow(\%friendClassHash, $currentClass, $sMethodMethodMatrix);
	  $sBriandCouplingMetrics->{$classKey}->{DMMEC} = SumOfRow(\%descendentClassHash, $currentClass, $sMethodMethodMatrix);	
	  $sBriandCouplingMetrics->{$classKey}->{OMMEC} = SumOfRow("All", $currentClass, $sMethodMethodMatrix)
	                                                  - SumOfRow(\%OthersDiffInverseFriendHash, $currentClass, $sMethodMethodMatrix);

   my %descendentClassHash;
   getDescendentClasses($currentClass, \%descendentClassHash);	                                                   
	 $sBriandCouplingMetrics->{totalNOD} += NOD($currentClass, \%descendentClassHash);  #ΪDOR�ļ�����׼��                                                 
	                                                   
	}#END for
	
	return 1;	
}#END sub BriandCouplingSeries


sub SumOfColumn{
	#����һ������A[i, j], ����j = 1..n ʱ A[i, j]���ۼӺ�, �������겻��, �б仯ʱ�ĺ�
	my $sRowClass = shift;
	my $sColumnClassHash = shift;
	my $sMatrixHash = shift;
	
	my $result = 0;	
	my $rowClassKey = getClassKey($sRowClass); 
	
	my $actualColumnHash;
	
	if ($sColumnClassHash =~ m/All/i){
		$actualColumnHash = $sMatrixHash->{Matrix}->{$rowClassKey};
	}
	else{
		$actualColumnHash = $sColumnClassHash;
	}
		
	foreach my $columnClassKey (keys %{$actualColumnHash}){				
		next if (!exists $sMatrixHash->{Matrix}->{$rowClassKey});
		next if (!exists $sMatrixHash->{Matrix}->{$rowClassKey}->{$columnClassKey});
		
		$result = $result + $sMatrixHash->{Matrix}->{$rowClassKey}->{$columnClassKey};
	}
		
	return $result;	
}#END sub SumOfColumn



sub SumOfRow{
	#����һ������A[i, j], ����i = 1..n ʱ A[i, j]���ۼӺ�, �������겻��, �б仯ʱ�ĺ�
	my $sRowClassHash = shift;
	my $sColumnClass = shift;
	my $sMatrixHash = shift;
	
	my $result = 0;
	my $columnClassKey = getClassKey($sColumnClass); 
	
	my $actualRowHash;
	
	if ($sRowClassHash =~ m/All/i){
		$actualRowHash = $sMatrixHash->{Matrix};
	}
	else{
		$actualRowHash = $sRowClassHash;
	}	
		
	foreach my $rowClassKey (keys %{$actualRowHash}){		
		next if (!exists $sMatrixHash->{Matrix}->{$rowClassKey});
		next if (!exists $sMatrixHash->{Matrix}->{$rowClassKey}->{$columnClassKey});
		
		$result = $result + $sMatrixHash->{Matrix}->{$rowClassKey}->{$columnClassKey};
	}
		
	return $result;	
}#END sub SumOfColumn


sub getUnionHash{
	my $firstHash = shift;
	my $secondHash = shift;
	my $unionHash = shift;
	
	foreach my $key (%{$firstHash}){
		$unionHash->{$key} = $firstHash->{$key};
	}
	
	foreach my $key (%{$secondHash}){
		$unionHash->{$key} = $secondHash->{$key};
	}
	
	return 1;
}


#$diffHash = $firstHash - $secondHash
sub getDiffHash{
	my $firstHash = shift;
	my $secondHash = shift;
	my $diffHash = shift;
	
	foreach my $key (%{$firstHash}){
		$diffHash->{$key} = $firstHash->{$key};
	}
	
	foreach my $key (%{$secondHash}){
		delete $diffHash->{$key} if (exists $diffHash->{$key});		
	}	
	
	return 1;
}


sub IFCAIC{
	my $sClass = shift;	
	my $sInverseFriendClassHash = shift;
	my $sClassAttributeMatrix = shift;
	
	print "\t\t\t computing IFCAIC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($sClass, $inverseFriend);
  }

	print "...IFCAIC END\n" if ($debug);
	  
	return $result;
}#END sub IFCAIC


sub ACAIC{
	my $sClass = shift;
	my $sAncestorClassHash = shift;
	
	print "\t\t\t computing ACAIC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sAncestorClassHash}){
  	my $ancestor = $sAncestorClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($sClass, $ancestor);
  }
  
  print "... ACAIC END\n" if ($debug);
  
	return $result;
}#END sub ACAIC



sub OCAIC{
	my $sClass = shift;
	my $sFriendClassHash = shift;
	my $sOtherClassHash = shift;
	
	print "\t\t\t computing OCAIC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($sClass, $other);
  }
 
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($sClass, $friend);
  }
  
  print "...OCAIC END\n" if ($debug);
	return $result;
}#END sub OCAIC



sub FCAEC{
	my $sClass = shift;
	my $sFriendClassHash = shift;
#	print "\t\t\t computing FCAEC...\n";
	
	my $result = 0;	
  
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($friend, $sClass);
  }
  
	return $result;
}#END sub FCAEC



sub DCAEC{
	my $sClass = shift;
	my $sDescendentClassHash = shift;
	
	print "\t\t\t computing DCAEC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sDescendentClassHash}){
  	my $descendent = $sDescendentClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($descendent, $sClass);
  }
  
  print "...DCAEC END\n" if ($debug);
  
	return $result;
}#END sub DCAEC



sub OCAEC{	
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	my $sOtherClassHash = shift;
	
	print "\t\t\t computing OCAEC..." if ($debug);
	
	my $result = 0;	
 
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($other, $sClass);
  }
  
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassAttributeInteraction($inverseFriend, $sClass);
  }
  
  print "...OCAEC END\n" if ($debug);
  
	return $result;
}#END sub OCAEC



sub IFCMIC{
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	
#	print "\t\t\t computing IFCMIC...\n";
	
	my $result = 0;	
  
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($sClass, $inverseFriend);
  }
  
	return $result;
}#END sub IFCMIC




sub ACMIC{
	my $sClass = shift;
	my $sAncestorClassHash = shift;
	print "\t\t\t computing ACMIC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sAncestorClassHash}){
  	my $ancestor = $sAncestorClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($sClass, $ancestor);
  }
  
  print "...ACMIC END\n" if ($debug);
	return $result;
}#END sub ACMIC



sub OCMIC{
	my $sClass = shift;
	my $sFriendClassHash = shift;
	my $sOtherClassHash = shift;	
	
	print "\t\t\t computing OCMIC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($sClass, $other);
  }  
  
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($sClass, $friend);
  }
  
  print "...OCMIC END\n" if ($debug);
	return $result;
}#END sub OCMIC



sub FCMEC{
	my $sClass = shift;
	my $sFriendClassHash = shift;
#	print "\t\t\t computing FCMEC...\n";
	
	my $result = 0;	
  
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($friend, $sClass);
  }
  
	return $result;
}#END sub FCMEC


sub DCMEC{
	my $sClass = shift;
	my $sDescendentClassHash = shift;
	
	print "\t\t\t computing DCMEC..." if ($debug);
	
	my $result = 0;
	
  foreach my $key (keys %{$sDescendentClassHash}){
  	my $descendent = $sDescendentClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($descendent, $sClass);
  }
  
  print "...DCMEC END\n" if ($debug);
	return $result;
}#END sub DCMEC



sub OCMEC{
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	my $sOtherClassHash = shift;
	
	print "\t\t\t computing OCMEC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfClassMethodInteraction($other, $sClass);
  }
  
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + 
    ($inverseFriend, $sClass);
  }
  
  print "...OCMEC END\n" if ($debug);
  
	return $result;
}#END sub OCMEC



sub IFMMIC{
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	
#	print "\t\t\t computing IFMMIC...\n";
	
	my $result = 0;
	
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($sClass, $inverseFriend);
  }
  
	return $result;
}#END sub IFMMIC


sub AMMIC{
	my $sClass = shift;
	my $sAncestorClassHash = shift;
	
	print "\t\t\t computing AMMIC..." if ($debug);
	
	my $result = 0;
  
  foreach my $key (keys %{$sAncestorClassHash}){
  	my $ancestor = $sAncestorClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($sClass, $ancestor);
  }
  
  print "...AMMIC END\n" if ($debug);
  
	return $result;
}#END sub AMMIC


sub OMMIC{	
	my $sClass = shift;
  my $sFriendClassHash = shift;
  my $sOtherClassHash	= shift;
	
	print "\t\t\t computing OMMIC..." if ($debug);
	
	my $result = 0;
 
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($sClass, $other);
  }
  
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($sClass, $friend);
  }
  
  print "...OMMIC END\n" if ($debug);
  
	return $result;
}#END sub OMMIC


sub FMMEC{
	my $sClass = shift;
	my $sFriendClassHash = shift;
	
#	print "\t\t\t computing FMMEC...\n";
	
	my $result = 0;
	
  foreach my $key (keys %{$sFriendClassHash}){
  	my $friend = $sFriendClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($friend, $sClass);
  }
  
	return $result;
}#END sub FMMEC



sub DMMEC{
	my $sClass = shift;
	my $sDescendentClassHash = shift;
	
	print "\t\t\t computing DMMEC..." if ($debug);
	
	my $result = 0;	
  
  foreach my $key (keys %{$sDescendentClassHash}){
  	my $descendent = $sDescendentClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($descendent, $sClass);
  }
  
  print "...DMMEC END\n" if ($debug);
  
	return $result;
}#END sub DMMEC


sub OMMEC{
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	my $sOtherClassHash = shift;	
	
	print "\t\t\t computing OMMEC..." if ($debug);
	
	my $result = 0;
	
  foreach my $key (keys %{$sOtherClassHash}){
  	my $other = $sOtherClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($other, $sClass);
  }
  
  foreach my $key (keys %{$sInverseFriendClassHash}){
  	my $inverseFriend = $sInverseFriendClassHash->{$key};  	  
    $result = $result + getNoOfMethodMethodInteraction($inverseFriend, $sClass);
  }
  
  print "...OMMEC END\n" if ($debug);
  
	return $result;
}#END sub OMMEC


sub CBI{
	my $sClass = shift;
	my $sDescendentHash = shift;
	
	print "\t\t\t computing CBI..." if ($debug);
	
	my $sumIMC = 0;
	
	my @methodArray = getEntsInClass($sClass, "define", "function ~unresolved ~unknown, method ~unresolved ~unknown");
	
	foreach my $func (@methodArray){
		$sumIMC = $sumIMC + IMC($func);
	}
	
	my $result = $sumIMC * NOD($sClass, $sDescendentHash);
	
	print "...CBI END\n" if ($debug);
	
	return $result;	
}#END sub CBI


sub CCAndAMC{
	my $sClass = shift;
	
	print "\t\t\t computing CC and AMC..." if ($debug);
	
	my $valueCC = 0;	
	
	my $currentClassName = getLastName($sClass->name());	
	
	my @methodArray = getEntsInClass($sClass, "Define","Function ~Unknown ~Unresolved, Method ~Unknown ~Unresolved");	
	
	return (0, 0) if (@methodArray == 0);
	
	foreach my $func (@methodArray){
#		print "\t func = ", $func->name(), "\n";
	
		#�������������õķǾֲ�����
		my @localVariableList = $func->refs("Use, Set, Modify", "Object ~unknown ~unresolved ~Local, Variable ~unknown ~unresolved ~Local");

		foreach my $variable (@localVariableList){	
			my $variableDefineInEnt = $variable->ent()->ref("Definein", "");		
		  if (!$variableDefineInEnt){  #ȫ�ֱ���
		  	$valueCC++;
		  	next;		  	
		  }	
		  
		  next if ($variableDefineInEnt->ent()->library() =~ m/Standard/i);
		  
		  my $variableDefineInEntName = getLastName($variableDefineInEnt->ent()->name);		  		  
		  
		  next if ($variableDefineInEntName eq $currentClassName); 		  

#		  print "\t\t\t non local variable = ", $variable->ent()->name(), "\n";
#		  print "\t\t\t type = ", $variableDefineInEntName, "\n";
#		  print "\t\t\t current = ", $currentClassName, "\n";
		  
		  $valueCC++;
		}
		
		
		#���������е��õķ���
		my @calledFuncSet = $func->refs("call", "function ~unresolved ~unknown, method ~unresolved ~unknown");
  	foreach my $calledfunc (@calledFuncSet){
  		my $calledFuncDefineInEnt = $calledfunc->ent()->ref("Definein", "");
  		if (!$calledFuncDefineInEnt){
  			$valueCC++;
  			next;  			
  		}  		 
  		
  		next if ($calledFuncDefineInEnt->ent()->library() =~ m/Standard/i);
  		
  		my $calledFuncDefineInEntName = getLastName($calledFuncDefineInEnt->ent()->name());		
  		next if ($calledFuncDefineInEntName eq $currentClassName); 
  		

#		  print "\t\t\t non local method = ", $calledfunc->ent()->name(), "\n";
#		  print "\t\t\t type = ", $calledFuncDefineInEntName, "\n";
#		  print "\t\t\t current = ", $currentClassName, "\n";  		
  		
  		$valueCC++;  		
	  }
	}
	
	my $valueAMC = $valueCC / @methodArray;
	
	print "...CCAndAMC END\n" if ($debug);
	
	return ($valueCC, $valueAMC);	
}#END sub CCAndAMC



sub UCL{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	my $sAncestorHash = shift;
	my $sDescendentHash = shift;
	
	my $currentClassKey = getClassKey($sClass);
	
	print "\t\t\t computing UCL..." if ($debug);
	
	my %ancestorHash = %{$sAncestorHash};
	my %descendentHash = %{$sDescendentHash};	#֮������Hash��, ���Ƕ�̳е����

	my $miu13 = 0;
	
	#������������(��������, ��������)Ϊ���͵����Ե���Ŀ
	my @attributeArray = getRefsInClass($sClass, "Define","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresovled");	
  
	foreach my $attribute (@attributeArray){
		my $attributeClass = $attribute->ent()->ref("Typed", "Class");		
		next if (!$attributeClass);		
		next if ($attributeClass->ent()->library() =~ m/Standard/i);
		
		#ȥ����׼����, ֻͳ��Ӧ����
		my $attributeClassKey = getClassKey($attributeClass->ent());
		#next if (!exists $sAllClassNameHash->{$attributeClassKey});
		
		next if ($currentClassKey eq $attributeClassKey); 		
 	  next if (exists $ancestorHash{$attributeClassKey});
		next if (exists $descendentHash{$attributeClassKey});
		
		$miu13++; 
	}
	
	
	#���㷽������������(��������, ��������)Ϊ���͵ľֲ�����(���������ͷ���ֵ)����Ŀ
	my @methodArray = getEntsInClass($sClass, "Define","Function ~Unknown ~Unresolved, Method ~Unknown ~Unresolved");	
	my $miu24 = 0;
	
	foreach my $func (@methodArray){
#		print "\t func = ", $func->name(), "\n";
		#������������
		my @parameterList = $func->ents("Define", "Parameter");
		
		foreach my $parameter (@parameterList){
			my $parameterClass = $parameter->ref("Typed", "Class");
			next if (!$parameterClass);		
			next if ($parameterClass->ent()->library() =~ m/Standard/i);

  		#ȥ����׼����, ֻͳ��Ӧ����
		  my $parameterClassKey = getClassKey($parameterClass->ent());		
		  #next if (!exists $sAllClassNameHash->{$parameterClassKey});
		  
		  next if ($currentClassKey eq $parameterClassKey); 
		  next if (exists $ancestorHash{$parameterClassKey});
		  next if (exists $descendentHash{$parameterClassKey});
		  
		  $miu24++;
		}
		
	
		#���������ж���ľֲ�����
		my @localVariableList = $func->refs("define", "Object ~Unknown ~Unresolved, Variable ~Unknown ~Unresolved");
		foreach my $variable (@localVariableList){	
			my $variableClass = $variable->ent()->ref("Typed", "Class");		
		  next if (!$variableClass);		
		  next if ($variableClass->ent()->library() =~ m/Standard/i);
		  
		  #ȥ����׼����, ֻͳ��Ӧ����		  
		  my $variableClassKey = getClassKey($variableClass->ent());		
		  #next if (!exists $sAllClassNameHash->{$variableClassKey});
		  
		  next if ($currentClassKey eq $variableClassKey); 
		  next if (exists $ancestorHash{$variableClassKey});
		  next if (exists $descendentHash{$variableClassKey});				  
		  
		  $miu24++;
		}
		
		
		#������������
		my $returnClass = $func->ref("Typed", "Class");
	  next if (!$returnClass);
	  next if ($returnClass->ent()->library() =~ m/Standard/i);
		
	  #ȥ����׼����, ֻͳ��Ӧ����
    my $returnClassKey = getClassKey($returnClass->ent());		
    #next if (!exists $sAllClassNameHash->{$returnClassKey});
    
    next if ($currentClassKey eq $returnClassKey); 
		next if (exists $ancestorHash{returnClassKey});
		next if (exists $descendentHash{returnClassKey});				
		
		$miu24++;  
	}
	
	
	my $result = $miu13 + $miu24;
	
	print "...UCL END\n" if ($debug);
	
	return $result;
}#END sub UCL


sub ICH{
	my $sClass = shift;
	
	print "\t\t\t computing ICH..." if ($debug);

	my @methodArray = getEntsInClass($sClass, "define","function ~unknown ~unresovled, method ~unknown ~unresovled");    
	
	my $result = 0;
  
  my $callingClassKey = getClassKey($sClass);


  #ֻ���Ƕ�̬�ĵ���    
  foreach my $method (@methodArray){
  	my $callingMethodName = getLastName($method->name());
  	my %polyCalledFuncSet;
  	PIM($method, \%polyCalledFuncSet);
  	
#  	print "\t\t calling method = ", $method->name(), "\n";
  	
  	foreach my $key (sort keys %polyCalledFuncSet){
  		my $calledFuncEnt = $polyCalledFuncSet{$key}->{funcEnt};
  		my $callCount = $polyCalledFuncSet{$key}->{callCount};
  		
  		my $calledClass = $calledFuncEnt->ref("Definein", "Class");
  		next if (!$calledClass);
  		
  		my $calledClassKey = getClassKey($calledClass->ent());  		
  		next if ($callingClassKey ne $calledClassKey);

  		#�ų���������
  		my $calledMethodName = getLastName($calledFuncEnt->name());
  		next if ($callingMethodName eq $calledMethodName);  		
  		
  		my @parameterSet = $calledFuncEnt->ents("Define", "Parameter");
  		
#  		print "\t\t\t called method = ", $key, "\n";
#  		print "\t\t\t count = ", $callCount, "\n";
  		$result = $result + (@parameterSet + 1) * $callCount;  
  	}
  }



#  #ֻ���Ǿ�̬�ĵ��� 
#  foreach my $method (@methodArray){
#  	my $callingMethodName = getLastName($method->name());
#  	my @calledFuncSet = $method->refs("call", "function  ~unresolved ~unknown, method  ~unresolved ~unknown");
#  	foreach my $func (@calledFuncSet){
#  		my $calledClass = $func->ent()->ref("Definein", "Class ~unknown ~unresovled");
#  		next if (!$calledClass);
#  		
#  		my $calledClassKey = getClassKey($calledClass->ent());  		
#  		next if ($callingClassKey ne $calledClassKey); 
#  		
#  		#�ų���������
#  		my $calledMethodName = getLastName($func->ent()->name());
#  		next if ($callingMethodName eq $calledMethodName);
#  		
#  		my @parameterSet = $func->ent()->ents("Define", "Parameter");
#  		$result = $result + @parameterSet + 1;  		
#  	}
#  }
  
  print "...ICH END\n" if ($debug);
  
  return $result;		
}#END sub ICH


sub SPoly{
	my $firstClass = shift;
	my $secondClass = shift;
	
	my %firstFuncHash;
	
	my @firstFuncList = getEntsInClass($firstClass, "Define", "Function ~private, Method ~private");
	foreach my $func (@firstFuncList){
		my $signature = getFuncSignature($func, 0);		
		my $realFuncName = getLastName($func->name());				
		$firstFuncHash{$realFuncName}{$signature} = 1;
	}
	
	my %secondFuncHash;
	
	my @secondFuncList = getEntsInClass($secondClass, "Define", "Function ~private, Method ~private");
	foreach my $func (@secondFuncList){
		my $signature = getFuncSignature($func, 0);
		
		my $realFuncName = getLastName($func->name());	

		$secondFuncHash{$realFuncName} = 1 if (exists $firstFuncHash{$realFuncName} && !exists $firstFuncHash{$realFuncName}{$signature});
	}

	my $result = 0;
	$result = (keys %secondFuncHash);

	return $result;
}#END sub SPoly



sub SPA{
	my $sClass = shift;
	my $sAncestorHash = shift;
	
#	print "\t\t\t computing SPA...\n";
	
	my %ancestorHash = %{$sAncestorHash};
  
	my $result = 0;
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		$result = $result + SPoly($ancestorClass, $sClass);
	}
	
	return $result;	
}#END sub SPA


sub SPD{
	my $sClass = shift;	
	my $sDescendentHash = shift;
	
#	print "\t\t\t computing SPD...\n";
	
	my %descendentHash = %{$sDescendentHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my $result = 0;
	foreach my $key (keys %descendentHash){
		my $descendentClass = $descendentHash{$key};
		$result = $result + SPoly($descendentClass, $sClass);		
	}
	
	return $result;	
}#END sub SPD


sub DPoly{
	my $firstClass = shift;
	my $secondClass = shift;

	my %firstFuncHash;
	
	my @firstFuncList = getEntsInClass($firstClass, "Define", "Function ~private, Method ~private");
	foreach my $func (@firstFuncList){
		my $signature = getFuncSignature($func, 1);			
		$firstFuncHash{$signature} = 1;
	}
	
	my %secondFuncHash;
	
	my @secondFuncList = getEntsInClass($secondClass, "Define", "Function ~private, Method ~private");
	foreach my $func (@secondFuncList){
		my $signature = getFuncSignature($func, 1);
		$secondFuncHash{$signature} = 1 if (exists $firstFuncHash{$signature});
	}
		
	my $result = 0;
	$result = (keys %secondFuncHash);
	
	return $result;
}#END sub DPoly


sub DPA{
	my $sClass = shift;
	my $sAncestorHash = shift;
	
#	print "\t\t\t computing DPA...\n";
	
	my %ancestorHash = %{$sAncestorHash};
	
	my $result = 0;
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		$result = $result + DPoly($ancestorClass, $sClass);
	}
	
	return $result;	
}#END sub DPA


sub DPD{
	my $sClass = shift;
	my $sDescendentHash = shift;
	
#	print "\t\t\t computing DPD...\n";
	
	my %descendentHash = %{$sDescendentHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my $result = 0;
	foreach my $key (keys %descendentHash){
		my $descendentClass = $descendentHash{$key};
		$result = $result + DPoly($descendentClass, $sClass);
	}
	
	return $result;	
}#END sub DPD


sub SP{
	my $sSPA = shift;
	my $sSPD = shift;	
	
	my $result = 0;
	$result = $sSPA + $sSPD;	
	
	return $result;
}#END sub SP


sub DP{
	my $sDPA = shift;
	my $sDPD = shift;
	
	my $result = 0;
	$result = $sDPA + $sDPD;		

	return $result;
}#END sub DP

sub CHM{
	my $sDIT = shift;
	my $sNOD = shift;
	my $sNOP = shift;
	my $sNMI = shift;
	my $sNMA = shift;
	
	my $result = 0;
	$result = $sDIT + $sNOD + $sNOP + $sNMI + $sNMA;
	
	return $result;	
}#END sub CHM


sub DOR{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	my $sDescendentHash = shift;
	my $sTrTr = shift;
	
  my $tt = scalar (keys %{$sAllClassNameHash});
  my $trtr = $sTrTr;
	
	my $rc;
	$rc = NOD($sClass, $sDescendentHash);
	
	my $result = 0;
	
	for (my $k = 1; $k <= $rc; $k++){
		$result = $result + $k/($tt + $trtr);
	}
	
	return $result;	
}#END sub DOR



sub OVO{
	my $sClass = shift;
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~private, Method ~private");
	
	my %funcHash;
	
	foreach my $func (@currentFuncList){
		$funcHash{$func->name()}++;		
	}
	
	my $result = 0;
	
	foreach my $key (keys %funcHash){
		$result = $result + $funcHash{$key} if $funcHash{$key} > 1;
	}
	
	return $result;	
}


sub NM{
	my $sClass = shift;
	my $sAncestorHash = shift;
	
	print "\t\t\t computing NM..." if ($debug);
	
	my %ancestorHash = %{$sAncestorHash};
	
	my %methodInAncestor; # �������еķ�����
	
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		
		my @funcList = getEntsInClass($ancestorClass, "Define", "Function ~private, Method ~private");
		
		foreach my $func (@funcList){
			my $signature = getFuncSignature($func, 1);
			$methodInAncestor{$signature} = 1;
		}
	}
	
	my $count = 0;
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function, Method");
	
	foreach my $func (@currentFuncList){
		my $currentSignature = getFuncSignature($func, 1);
		if (exists $methodInAncestor{$currentSignature}){
			$count++;
#			print "count = ", $count, " method = ", $currentSignature, "\n";
		};		
	}
	
#	print "methodInAncestor = ", scalar (keys %methodInAncestor), "\n";
#	print "currentFuncList = ", scalar @currentFuncList, "\n";	
#	print "count = ", $count, "\n";
	
	my $result = 0;

	$result = (keys %methodInAncestor) + @currentFuncList - $count;

	
	print "...NM END\n" if ($debug);
		
	return $result;
}#END sub NM


sub NA{
	my $sClass = shift;
	my $sAncestorHash = shift;
	
	print "\t\t\t computing NA..." if ($debug);
	
	my %ancestorHash = %{$sAncestorHash};
	
	my $result = $sClass->metric("CountDeclClassVariable") + $sClass->metric("CountDeclInstanceVariable");;
	
	
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};	
		my @attributeArray = getEntsInClass($ancestorClass, "define", "member object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");      
		$result = $result + scalar @attributeArray;		
	}
	
	return $result;
}#END sub NA


sub Nmpub{
	my $sClass = shift;

	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~private ~protected ~unresolved, Method ~private ~protected  ~unresolved");	
	my $result = 0;
	$result = @currentFuncList;
	
	return $result;	
}#END sub Nmpub



sub NMNpub{
	my $sClass = shift;

	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~unresolved, Method ~unresolved");	
	my $result = 0;
	$result = @currentFuncList - Nmpub($sClass);
	
	return $result;	
}#END sub NMNpub



sub NumPara{
	my $sClass = shift;
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~unresolved, Method ~unresolved");	
	
	my $result = 0;
	
	foreach my $func (@currentFuncList){
		my @parameterList = $func->ents("Define", "Parameter");
		$result = $result + @parameterList;
	}
	
	return $result;	
}#END sub NumPara



sub InheritanceSeries{
	my $sClass = shift;
	my $sAllClassNameHash = shift;
	my $sAncestorHash = shift;
	my $sAncestorLevel = shift;
	my $sDescendentClassHash = shift;
	my $sHashMetrics = shift;
	
	my $valueNOC = NOC($sClass);
	$sHashMetrics->{Inheritance}->{NOC} = $valueNOC;
	
	my $valueNOP = NOP($sClass);
	$sHashMetrics->{Inheritance}->{NOP} = $valueNOP;
	
	my $valueDIT = DIT($sClass);
	$sHashMetrics->{Inheritance}->{DIT} = $valueDIT;
	
	my $valueAID = AID($sClass);
	$sHashMetrics->{Inheritance}->{AID} = $valueAID;
	
	my $valueCLD = CLD($sClass);
	$sHashMetrics->{Inheritance}->{CLD} = $valueCLD;
	
	# my $start = getTimeInSecond();
	my $valueNOD = NOD($sClass, $sDescendentClassHash);
	$sHashMetrics->{Inheritance}->{NOD} = $valueNOD;
	# reportComputeTime($start, "NOD");  
	
	# $start = getTimeInSecond();
	my $valueNOA = NOA($sClass, $sAncestorHash);
	$sHashMetrics->{Inheritance}->{NOA} = $valueNOA;
	# reportComputeTime($start, "NOA");  
	
	# $start = getTimeInSecond();
	my $valueNMO = NMO($sClass, $sAncestorHash);
	$sHashMetrics->{Inheritance}->{NMO} = $valueNMO;
	# reportComputeTime($start, "NMO");  
	
	# $start = getTimeInSecond();
	my $valueNMI = NMI($sClass, $sAncestorHash);
	$sHashMetrics->{Inheritance}->{NMI} = $valueNMI;
	# reportComputeTime($start, "NMI"); 	
	
	# $start = getTimeInSecond();
	my $valueNMA = NMA($sClass);
	$sHashMetrics->{Inheritance}->{NMA} = $valueNMA;
	# reportComputeTime($start, "NMA"); 
	
	
	my $valueSIX = SIX($valueNMO, $valueNMA, $valueNMI, $valueDIT);
	$sHashMetrics->{Inheritance}->{SIX} = $valueSIX;
	
	
	# $start = getTimeInSecond();
	my $valueNPBM = getNoOfPBRdM($sClass, $sAncestorHash, $sAncestorLevel); #�õ�������Ϊ��override������Ŀ  Preserved Behavior OverRide Method
	my $valuePII = PII($valueNPBM, $valueNMO, $valueNMA, $valueNMI, $valueDIT);
	$sHashMetrics->{Inheritance}->{PII} = $valuePII;
	# reportComputeTime($start, "PII"); 
	
	# $start = getTimeInSecond();
	my $valueSPA = SPA($sClass, $sAncestorHash);
	$sHashMetrics->{Inheritance}->{SPA} = $valueSPA;
	# reportComputeTime($start, "SPA"); 


  # $start = getTimeInSecond();
	my $valueSPD = SPD($sClass, $sDescendentClassHash);
	$sHashMetrics->{Inheritance}->{SPD} = $valueSPD;
	# reportComputeTime($start, "SPD"); 
	
	# $start = getTimeInSecond();
	my $valueDPA = DPA($sClass, $sAncestorHash);
	$sHashMetrics->{Inheritance}->{DPA} = $valueDPA;
	# reportComputeTime($start, "DPA"); 
	
	# $start = getTimeInSecond();
	my $valueDPD = DPD($sClass, $sDescendentClassHash);
	$sHashMetrics->{Inheritance}->{DPD} = $valueDPD;
	# reportComputeTime($start, "DPD"); 
	
	my $valueSP = SP($valueSPA, $valueSPD);
	$sHashMetrics->{Inheritance}->{SP} = $valueSP;
	
	my $valueDP = DP($valueDPA, $valueDPD);
	$sHashMetrics->{Inheritance}->{DP} = $valueDP;
	
	
#	my $valueCHM = CHM($valueDIT, $valueNOD, $valueNOP, $valueNMI, $valueNMA);
#	$sHashMetrics->{Inheritance}->{CHM} = $valueCHM;	
}#END sub InheritanceSeries


sub NOC{
	my $sClass = shift;
	
	#my $result = $sClass->metric("CountClassDerived");
	my @sonList = $sClass->refs("Derive, Extendby", "class", 1); #�������1, �ڷ���Java������ʱ����,ԭ��δ֪
	
	my $result = @sonList;	
	
	return $result;	
}#END sub NOC


sub NOP{
	my $sClass = shift;	
	
	my @parentList = $sClass->refs("Base, Extend", "class", 1); #�������1, �ڷ���Java������ʱ����,ԭ��δ֪
	
	my $result = @parentList;	
#	$result = $sClass->metric("CountClassBase");  #����Java, �������а�����ʵ�ֵĽӿ�, ���Բ����Զ���ļ���
	
	return $result;
}#END sub NOP
	

sub DIT{
	my $sClass = shift;
	#��Java��, �κ��඼��Object�ĺ�����, ������ֻ����Ӧ����ļ̳в��
	
#	return $sClass->metric("MaxInheritanceTree");

	my @parentList;
	
	foreach my $parent ($sClass->refs("Base, Extend", "class", 1)){
		push @parentList, $parent->ent();		
	}	
	
	return 0 if (!@parentList);
	
	my $result = 0;
	
	foreach my $parent (@parentList){
		my $tempDIT = DIT($parent);
		$result = $tempDIT if ($result < $tempDIT);
	}
	
	$result = $result + 1;	
	
  return $result;
} #END sub DIT


sub AID{
	my $sClass = shift;	
	
#	print "\t\t\t computing AID...";

	my @parentList;
	
	foreach my $parent ($sClass->refs("Base, Extend", "class", 1)){
		push @parentList, $parent->ent();		
	}	
	
	return 0 if (!@parentList);

	my $result = 0;
	
	foreach my $parent (@parentList){
		$result = $result + AID($parent);
	}
	
	$result = $result / (scalar @parentList) + 1;
	
#	print "...AID END\n";
	
	return $result;	
}#END sub AID


sub CLD{
	my $sClass = shift;	

#  print "\t\t\t computing CLD...";

	my @sonList;
	
	foreach my $son ($sClass->refs("Derive, Extendby", "class", 1)){
		push @sonList, $son->ent();		
	}	
	
	return 0 if (!@sonList);
	
	my $result = 0;
	
	foreach my $son (@sonList){
		$result = CLD($son) if ($result < CLD($son));
	}
		
	$result = $result + 1;
	
#	print "...CLD END\n";
	
	return $result;	
}#END sub CLD



sub NOD{
	my $sClass = shift;	
	my $sDescendentHash = shift;
	
	my %descendentHash = %{$sDescendentHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my $result = 0;
	$result = (keys %descendentHash); 
	return $result;
}#END sub NOD


sub NOA{
	my $sClass = shift;	
	my $sAncestorHash = shift;
	
	my %ancestorHash = %{$sAncestorHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my $result = 0;
	$result = (keys %ancestorHash);
	
	return $result;	
}#END sub NOA


sub getFuncSignature{
	my $func = shift;
	my $includeReturnType = shift; 
	
	my $signature;
  my @wordList = split /\./, $func->name();
  
  if ($includeReturnType){
	  $signature = $func->type()." ".$wordList[$#wordList]."(";
	}
	else{
		$signature = $wordList[$#wordList]."(";
	} 
		
	my $first = 1;
	foreach my $param ($func->ents("Define", "Parameter")){
		$signature = $signature."," unless $first;
		$signature = $signature.$param->type();			
		$first = 0;
	}		
	
	$signature = $signature.")";
	
	return $signature;	
}#END sub getFuncSignature

sub getAttributeSignature{
	#����+"::"+������
	my $sClass = shift;
	my $attribute = shift;
	
	my $signature = getLastName($sClass->name())."::".getLastName($attribute->name());
	
	return $signature;	
}#END sub getFuncSignature


sub NMO{
	my $sClass = shift;	
	my $sAncestorHash = shift;
	
	print "\t\t\t computing NMO..." if ($debug);
	
	my %ancestorHash = %{$sAncestorHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my %methodInAncestor; # �������еķ�����
	
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		
		my @funcList = getEntsInClass($ancestorClass, "Define", "Function ~private,Method ~private");
		
		foreach my $func (@funcList){
			my $signature = getFuncSignature($func, 1);
			$methodInAncestor{$signature} = 1;
		}
	}
	
	my $result = 0;
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~private ~unresolved,Method ~private ~unresolved");
	
	foreach my $func (@currentFuncList){
		my $currentSignature = getFuncSignature($func, 1);
		$result++ if (exists $methodInAncestor{$currentSignature});		
	}
		
	print "...NMO END\n" if ($debug);
	
	return $result;
}#END sub NMO


sub NMI{
	my $sClass = shift;	
	my $sAncestorHash = shift;
	
	print "\t\t\t computing NMI..." if ($debug);
	
	my %ancestorHash = %{$sAncestorHash};	#֮������Hash��, ���Ƕ�̳е����
	
	my %methodInAncestor; # �������еķ�����
	
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		
		my @funcList = getEntsInClass($ancestorClass, "Define", "Function ~private,Method ~private");
		
		foreach my $func (@funcList){
			my $signature = getFuncSignature($func, 1);
			$methodInAncestor{$signature} = 1;
		}
	}
	
	my $count = 0;
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function ~private ~unresolved,Method ~private ~unresolved");
	
	foreach my $func (@currentFuncList){
		my $currentSignature = getFuncSignature($func, 1);
		$count++ if (exists $methodInAncestor{$currentSignature});		
	}
		
	my $result = (keys % methodInAncestor) - $count;
	
	print "...NMI END\n" if ($debug);
			
	return $result;
}#END sub NMI


sub NMA{
	my $sClass = shift;	
	
  print "\t\t\t computing NMA..." if ($debug);	

	my %addedMethodHash;	
	getAddedMethods($sClass, \%addedMethodHash);
	
	my $result = 0;
	$result = (keys %addedMethodHash);	
	
	print "...NMA END\n" if ($debug);
		
	return $result;
}#END sub NMA



sub SIX{
	my $sNMO = shift;
	my $sNMA = shift;
	my $sNMI = shift;
	my $sDIT = shift;
	
	return 0 if (($sNMO + $sNMA + $sNMI) == 0);
	
	my $result = 0;
	$result = $sNMO * $sDIT / ($sNMO + $sNMA + $sNMI);
	
	return $result;
}#END sub SIX


sub PII{
  my $sNPBM = shift; 
	my $sNMO = shift;
	my $sNMA = shift;
	my $sNMI = shift;
	my $sDIT = shift;
	
	my $valuePP = 0;	
	if ($sNMO > 0){
		$valuePP = ($sNMO - $sNPBM) / $sNMO;
	}
	
	my $valueOO = 0;	
	if ($sNMO + $sNMI > 0){
		$valueOO = $sNMO / ($sNMO + $sNMI);
	}

	my $valueNN = 0;	
	$valueNN = $sDIT / ($sDIT + 1) * $sNMA / ($sNMA + 1);
	
	my $valuePII = ($valuePP + $valueOO + $valueNN) / 3;

	return $valuePII;
}#END sub PII


sub getNoOfPBRdM{
	my $sClass = shift;
	my $sAncestorHash = shift;
	my $sAncestorLevel = shift;	
	
	my $valueNPBRdM = 0; 

	my @methodArray = getEntsInClass($sClass, "define", "function ~private ~unknown ~unresolved, method ~private ~unknown ~unresolved");
	foreach my $localFunc (@methodArray){
		my $localSignature = getFuncSignature($localFunc, 1);
		
		my $find = 0; #�ҵ�overriden�ķ���?
	
	  #���������в��ұ�overiding�ķ���
	  
	  FINDLOOP:  #��ĳ�����������ҵ���ͬ�����ķ���, ������������ѭ��
	  foreach my $level (sort keys %{$sAncestorLevel}){
		  my %ancestorKeyHash = %{$sAncestorLevel->{$level}};
		  
		  foreach my $classKey (keys %ancestorKeyHash){
			  my $ancestorClass = $sAncestorHash->{$classKey};			
			  my @ancestorMethodArray = getEntsInClass($ancestorClass, "define", "function ~private ~unknown, method ~private ~unknown");
			  
			  foreach my $ancestorFunc (@ancestorMethodArray){
				  my $ancestorSignature = getFuncSignature($ancestorFunc, 1);
				  
				  if ($ancestorSignature eq $localSignature){ #����ҵ�, �����	
				  	my $temp = isBehaviorPreserved($sClass, $localFunc, $ancestorClass, $ancestorFunc);	  	
#				  	print "\t\t\t Behavior Preserved? = ", $temp, "\n";
				  	$valueNPBRdM = $valueNPBRdM + $temp;				  	
				  	$find = 1;
				  	last FINDLOOP;		#��ĳ�����������ҵ���ͬ�����ķ���, ������������ѭ��		  	
				  }#END if				  
				}#END for				
			}
			
		}
	}
				  
  return $valueNPBRdM;
}#END sub getNoOfPBRdM


sub isBehaviorPreserved{
	#�����������Ϊ��behavior preserved: (1)������ķ����ǿշ���; (2)��ǰ��ķ���������������ķ���, ��������Ϊ��չ
	my $sClass = shift;
	my $sLocalFunc = shift;
	my $sAncestorClass = shift;
	my $sAncestorFunc = shift;
	
	#�ж�������ķ�����û�п�ִ�����. ע��: һ��Ҫ��"CountStmtExe"
	#������"CountStmt". ԭ��: Java�ķ�����ʹû����, metric("CountStmt")������1
	return 1 if (!$sAncestorFunc->metric("CountStmtExe")); 
	
  my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($sLocalFunc);
  return 0 if ($lexer eq "undef");
  
	my @calledMethodSet = $sLocalFunc->refs("call", "function ~unknown ~unresolved, method ~unknown ~unresolved");
	my $ancestorFuncCalled = 0;	
	
	my $localFuncSignature = getFuncSignature($sLocalFunc, 1);
		
	my $i = 0;		
	while ($i < @calledMethodSet && !$ancestorFuncCalled){		
		my $calledFunc = $calledMethodSet[$i]->ent();
		$i++;
		
		next if (getFuncSignature($calledFunc, 1) ne $localFuncSignature); #�����ò�ͬ�����ķ���, ����
		
		my $calledClass = $calledFunc->ref("Definein", "class");
		next if (!$calledClass);
		
		next if (getClassKey($calledClass->ent()) ne getClassKey($sAncestorClass));
		
		$ancestorFuncCalled = 1;
	}	  
	
	return 1 if ($ancestorFuncCalled);
			
	return 0;
}#END sub isBehaviorPreserved


sub MI{
	my $sClass = shift;
	
	my $MI = "undef";
	
  my @methodArray = getRefsInClass($sClass, "define","function ~unknown ~unresolved,  method ~unknown ~unresolved");    
  my $classFuncCount = scalar(@methodArray);    
  
  my $classLineCount = 0;
  my $classComplexitySum = 0;
  
  my %class_metric = ();
  
  foreach my $method (@methodArray){
  	my $func = $method->ent();
  	
		my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);
		next if ($lexer eq "undef");
	  
	  my ($n1, $n2, $N1, $N2) = scanEntity($lexer,$startLine,$endLine);
	  
	  # do the calcs
	  my ($n, $N) = ($n1 + $n2, $N1 + $N2);
	 	
	 	#avoid log of 0 error    
    $n = 1 if ($n <= 0);
        
	  my $V = $n ? $N * ((log $n)/(log 2)) : 0;
	
    #Sum data for class
	  $classLineCount  += $func->metric("CountLine");	  
	  $classComplexitySum += $func->metric("CyclomaticStrict");
	
	  # add them to the class-based metrics
	  $class_metric{V} += $V;
   }  
    
        
   # if this class has functions defined, report totals for the class
   if (@methodArray > 0){
    	
    	#**********compute MI****************
      my ($avG, $avV, $avLoc, $perCM);
   
      #calculate average V, make it 1 if 0 to avoid log error.     
      if ($class_metric{V} == 0){
        $avV = 1;
      }
      else{
        $avV = $class_metric{V} / $classFuncCount;
      }  
        
      $avG = $classComplexitySum / $classFuncCount;
      $avLoc = $classLineCount / $classFuncCount;
      $perCM = $sClass->metric("RatioCommentToCode")*100;
               
      if ($avLoc == 0){
        $avLoc = 1;          	
      }                    

      
	    $MI =  171-5.2 * log($avV)-.23*$avG-16.2*log($avLoc) + 50 * sin(sqrt(2.4 * $perCM));  		
   }

   return $MI;
} # END sub MI


sub LCOMSeriesAndCAC{
	my $sClass = shift;
	my $sHashMetrics = shift; 
   
  my %AttributeReadTable = ();
  my %AttributeWriteTable = ();
  my %AttributeModifyTable = ();
  my %MethodWithoutAttributeParaTable = ();
  my %AttributeWithoutAccessTable = ();
  my %DirectCallMethodSet = ();  
   
  if (buildAttributeHashTables($sClass, 0, 0, 0, 0,
            \%AttributeReadTable, \%AttributeWriteTable, \%AttributeModifyTable, 
            \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable, \%DirectCallMethodSet)){
   
      my %noForMethod = ();
      
      my $attributeMethodMatrix = hashTable2Matrix(\%noForMethod, \%AttributeReadTable, \%AttributeWriteTable, 
                           \%AttributeModifyTable, \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable);
                           
      $sHashMetrics->{Cohesion}->{CAC} = CAC($attributeMethodMatrix);                     
   
      my @methodMethodMatrix = generateMethodMethodMatrix($attributeMethodMatrix, \%noForMethod, 0, \%DirectCallMethodSet);

      $sHashMetrics->{Cohesion}->{LCOM1} = LCOM1(\@methodMethodMatrix);
      $sHashMetrics->{Cohesion}->{LCOM2} = LCOM2(\@methodMethodMatrix);
      $sHashMetrics->{Cohesion}->{LCOM3} = LCOM3(\@methodMethodMatrix);       
      $sHashMetrics->{Cohesion}->{LCOM5} = LCOM5($attributeMethodMatrix);
#      $sHashMetrics->{Cohesion}->{NewLCOM5} = NewLCOM5($attributeMethodMatrix);
       
      my @methodMethodMatrix2 = generateMethodMethodMatrix($attributeMethodMatrix, \%noForMethod, 1, \%DirectCallMethodSet);
              
      $sHashMetrics->{Cohesion}->{LCOM4} = LCOM3(\@methodMethodMatrix2);       
      $sHashMetrics->{Cohesion}->{Co} = Co(\@methodMethodMatrix2);
#      $sHashMetrics->{Cohesion}->{NewCo} = NewCo(\@methodMethodMatrix2);
  }  
  
  $sHashMetrics->{Cohesion}->{LCOM6} = LCOM6($sClass);
} #End sub LCOMSeries



sub TCCLCCSeries{
	my $sClass = shift;
	my $sHashMetrics = shift; 
   
  my %AttributeReadTable = ();
  my %AttributeWriteTable = ();
  my %AttributeModifyTable = ();
  my %MethodWithoutAttributeParaTable = ();
  my %AttributeWithoutAccessTable = ();
  my %DirectCallMethodSet = ();  	

  if (buildAttributeHashTables($sClass, 1, 1, 0, 1,
             \%AttributeReadTable, \%AttributeWriteTable, \%AttributeModifyTable, 
             \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable, \%DirectCallMethodSet)){    
          
      my %noForMethod = ();

      my $attributeMethodMatrix = hashTable2Matrix(\%noForMethod, \%AttributeReadTable, \%AttributeWriteTable, 
                           \%AttributeModifyTable, \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable);

                        
      my @methodMethodMatrix = generateMethodMethodMatrix($attributeMethodMatrix, \%noForMethod, 0, \%DirectCallMethodSet);

      $sHashMetrics->{Cohesion}->{TCC} = TCC(\@methodMethodMatrix);
      $sHashMetrics->{Cohesion}->{LCC} = LCC(\@methodMethodMatrix);
      
      my %indirectCallByMethodSet = getIndirectCallByMethodSet(\%DirectCallMethodSet);
      my @methodMethodMatrix3 = generateMethodMethodMatrix($attributeMethodMatrix, \%noForMethod, 2, \%indirectCallByMethodSet);
      
      $sHashMetrics->{Cohesion}->{DCd} = TCC(\@methodMethodMatrix3);
      $sHashMetrics->{Cohesion}->{DCi} = LCC(\@methodMethodMatrix3);            
   }	
	
} # End sub TCCLCCSeries



sub CBMCSeries{
	my $sClass = shift;
	my $sHashMetrics = shift;

  my %AttributeReadTable = ();
  my %AttributeWriteTable = ();
  my %AttributeModifyTable = ();
  my %MethodWithoutAttributeParaTable = ();
  my %AttributeWithoutAccessTable = ();
  my %DirectCallMethodSet = ();  

   
  if (buildAttributeHashTables($sClass, 0, 1, 1, 1,
             \%AttributeReadTable, \%AttributeWriteTable, \%AttributeModifyTable, 
             \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable, \%DirectCallMethodSet)){                           
             	
      my %noForMethod = ();
      
      my ($attributeMethodMatrix, $noOfAttributes, $noOfMethods) = hashTable2Matrix(\%noForMethod, \%AttributeReadTable, \%AttributeWriteTable, 
                           \%AttributeModifyTable, \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable);
      
      if ($noOfMethods == 0 && $noOfAttributes > 1 
        || $noOfMethods > 1 && $noOfAttributes == 0){ 
        	$sHashMetrics->{Cohesion}->{CBMC} = 0;
        	$sHashMetrics->{Cohesion}->{ICBMC} = 0;
        }
      else{     
        	# $sHashMetrics->{Cohesion}->{CBMC} = CBMC($attributeMethodMatrix);
        	# $sHashMetrics->{Cohesion}->{ICBMC} = ICBMC($attributeMethodMatrix);
        }
    }
} # End sub CBMCSeries


sub OCCAndPCC{
	my $sClass = shift;
	my $sHashMetrics = shift;   
	
	my %AttributeReadTable = ();
  my %AttributeWriteTable = ();
  my %AttributeModifyTable = ();
  my %MethodWithoutAttributeParaTable = ();
  my %AttributeWithoutAccessTable = ();
  my %DirectCallMethodSet = (); 

  if (buildAttributeHashTables($sClass, 0, 0, 0, 1,
             \%AttributeReadTable, \%AttributeWriteTable, \%AttributeModifyTable, 
             \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable, \%DirectCallMethodSet)){                           
   
      my %noForMethod = ();
      
      my $attributeMethodMatrix = hashTable2Matrix(\%noForMethod, \%AttributeReadTable, \%AttributeWriteTable, 
                           \%AttributeModifyTable, \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable);
       
      my @methodMethodMatrix = generateMethodMethodMatrix($attributeMethodMatrix, \%noForMethod, 0, \%DirectCallMethodSet);
      
      $sHashMetrics->{Cohesion}->{OCC} = OCC(\@methodMethodMatrix);
      $sHashMetrics->{Cohesion}->{PCC} = PCC(\%AttributeReadTable, \%AttributeWriteTable, 
                 \%AttributeModifyTable, \%MethodWithoutAttributeParaTable);            
   }
} #End sub OCCandPCC


sub CAMCSeries{	
	my $sClass = shift;
	my $sHashMetrics = shift;	
  
  my %ParaTable = ();
	 
  if (buildParameterHashTable($sClass, 0, 0, 0, 0, \%ParaTable)){ #������һ������
      my @testParaTable = (keys %ParaTable);         
      
      if (scalar @testParaTable == 1 && $testParaTable[0] eq "withoutParameterAndAttribute"){      	
#   	    $sHashMetrics->{Cohesion}->{CAMCs} = 1;
#   	    $sHashMetrics->{Cohesion}->{NHDs} = 1;
#   	    $sHashMetrics->{Cohesion}->{SNHDs} = 1;   	    
      }
      else{#������һ�������в���      	
      	my %noForMethod = ();
      	
   	    my $parameterTypeMethodMatrix = hashTable2Matrix(\%noForMethod, \%ParaTable);    
   	     
   	    my ($CAMC, $CAMCs) = CAMC($parameterTypeMethodMatrix);
   	    my ($NHD, $NHDs) = NHD($parameterTypeMethodMatrix);
   	    my $SNHD = SNHD($parameterTypeMethodMatrix);  	
   	    my $SNHDs = SNHDs($parameterTypeMethodMatrix);  
   	     
   	    $sHashMetrics->{Cohesion}->{CAMC} = $CAMC;
#   	    $sHashMetrics->{Cohesion}->{CAMCs} = $CAMCs;
   	    $sHashMetrics->{Cohesion}->{NHD} = $NHD;
#   	    $sHashMetrics->{Cohesion}->{NHDs} = $NHDs;
   	    $sHashMetrics->{Cohesion}->{SNHD} = $SNHD;
#   	    $sHashMetrics->{Cohesion}->{SNHDs} = $SNHDs;
   	  }
   	}
   

#   if (buildParameterHashTable($sClass, 0, 0, 0, 1, \%ParaTable)){                               	
#      
#      my @testParaTable = (keys %ParaTable);   
#      
#      if (scalar @testParaTable == 1 && $testParaTable[0] eq "withoutParameterAndAttribute"){
#      	# ʲô��������, ��Ϊ��Ӧ������Ĭ��ֵ��undefined       
#   	    $sHashMetrics->{Cohesion}->{iCAMCs} = 1;
#   	    $sHashMetrics->{Cohesion}->{iNHDs} = 1;      	
#   	    $sHashMetrics->{Cohesion}->{iSNHDs} = 1; 
#      }
#      else{
#      	my %noForMethod = ();
#      	
#   	    my $parameterTypeMethodMatrix = hashTable2Matrix(\%noForMethod, \%ParaTable);    
#   	     
#   	    my ($iCAMC, $iCAMCs) = CAMC($parameterTypeMethodMatrix);
#   	    my ($iNHD, $iNHDs) = NHD($parameterTypeMethodMatrix);
#   	    my $iSNHD = SNHD($parameterTypeMethodMatrix);  	   	    
#   	    my $iSNHDs = SNHDs($parameterTypeMethodMatrix);  	   	    
#   	    
#   	    $sHashMetrics->{Cohesion}->{iCAMC} = $iCAMC;
#   	    $sHashMetrics->{Cohesion}->{iCAMCs} = $iCAMCs;
#   	    $sHashMetrics->{Cohesion}->{iNHD} = $iNHD;
#   	    $sHashMetrics->{Cohesion}->{iNHDs} = $iNHDs;
#   	    $sHashMetrics->{Cohesion}->{iSNHD} = $iSNHD;
#   	    $sHashMetrics->{Cohesion}->{iSNHDs} = $iSNHDs;
#   	  }
#   	}
} # Endsub CAMCSeries


sub SCOM{
	#Ŀǰֻ��C++������Ч
	#��Java�����ƺ��е�����, ���ص�"undef"̫��
	my $sClass = shift;
	
	print "\t\t\t computing SCOM..." if ($debug);
	
	my %AttributeReadTable = ();
  my %AttributeWriteTable = ();
  my %AttributeModifyTable = ();
  my %MethodWithoutAttributeParaTable = ();
  my %AttributeWithoutAccessTable = ();
  my %DirectCallMethodSet = ();  
   
  if (buildAttributeHashTables($sClass, 0, 0, 1, 1,
            \%AttributeReadTable, \%AttributeWriteTable, \%AttributeModifyTable, 
            \%MethodWithoutAttributeParaTable, \%AttributeWithoutAccessTable, \%DirectCallMethodSet)){   
            	
      my %methodAttributeHashTable; #������¼ÿ������ֱ�ӻ��߼�ӷ��ʵ�����
      my %attributeHashTable;  #����ͳ�����Ե���Ŀ
      
      foreach my $attributeKey (keys %AttributeReadTable){
      	my %tempMethodHash = %{$AttributeReadTable{$attributeKey}};
      	$attributeHashTable{$attributeKey} = 1;
      	foreach my $methodKey (keys %tempMethodHash){
      		$methodAttributeHashTable{$methodKey}->{$attributeKey} = 1;
      	}
      }
            	
      foreach my $attributeKey (keys %AttributeWriteTable){
      	my %tempMethodHash = %{$AttributeWriteTable{$attributeKey}};
      	$attributeHashTable{$attributeKey} = 1;
      	foreach my $methodKey (keys %tempMethodHash){
      		$methodAttributeHashTable{$methodKey}->{$attributeKey} = 1;
      	}
      }
            	
      foreach my $attributeKey (keys %AttributeModifyTable){
      	my %tempMethodHash = %{$AttributeModifyTable{$attributeKey}};
      	$attributeHashTable{$attributeKey} = 1;
      	foreach my $methodKey (keys %tempMethodHash){
      		$methodAttributeHashTable{$methodKey}->{$attributeKey} = 1;
      	}
      }           	      
     
      my $noOfAttributes = (keys %attributeHashTable); #������Ŀ, ֻͳ�Ʊ��������ʵ�����
      my $noOfMethods = (keys %methodAttributeHashTable); #������Ŀ, ֻͳ�Ʒ������Եķ���
      
      return 1 if ($noOfMethods == 1);
      return 0 if ($noOfAttributes < 1);
      
      my @methodAttributeTable; 
      
      my $ii = 0;
      foreach my $key (keys %methodAttributeHashTable){
      	$methodAttributeTable[$ii] = $methodAttributeHashTable{$key};
      	$ii++;
      }
      

#      for (my $i = 0; $i < $noOfMethods; $i++){
#      	print "\t\t i = ", $i, "\n";
#      	my %tempHash = %{$methodAttributeTable[$i]};
#      	foreach my $att (keys %tempHash){
#      		print "\t\t\t attribute = ", $att, "\n";
#      	}
#      }
      
      
      my $sum = 0; 
      
      for (my $i = 0; $i < $noOfMethods - 1; $i++){
      	for (my $j = $i + 1; $j < $noOfMethods; $j++){
      		my $Cij = CardIntersection($methodAttributeTable[$i],$methodAttributeTable[$j]); #����ǿ��
      		
      		if ($Cij){
      			my $min = (keys %{$methodAttributeTable[$i]});
      			$min = (keys %{$methodAttributeTable[$j]}) if ($min > (keys %{$methodAttributeTable[$j]}));
      			
      			$Cij = $Cij / $min;
      		}
      		
      		my $Wij;   #��Ӧ��Ȩֵ 
      		$Wij = CardUnion($methodAttributeTable[$i],$methodAttributeTable[$j]) / $noOfAttributes;
      		
      		$sum = $sum + $Cij * $Wij;
      		
#      		print "\t\t i = ", $i, "\t j= ", $j, ": \t";
#      		print "C = ", $Cij, "; \t";
#      		print "W = ", $Wij, "\n";
      	}
      }
      
      my $result = 2 * $sum / ($noOfMethods * ($noOfMethods - 1));
           
      print "...SCOM END\n" if ($debug);
      
      return $result;
  }	
  
  print "...SCOM END\n" if ($debug);
  return "undef";
}#END sub SCOM


sub CardIntersection{
	#��������Hash��, ��������key��ͬ����Ŀ
	my $sHashTableOne = shift;
	my $sHashTableTwo = shift;
	
	my $smallHashTable;
	my $largeHashTable;
	
	if ((keys %{$sHashTableOne}) > (keys %{$sHashTableTwo})){
		$smallHashTable = $sHashTableTwo;
		$largeHashTable = $sHashTableOne;
	}
	else{
		$smallHashTable = $sHashTableOne;
		$largeHashTable = $sHashTableTwo;
	}
		
	my $result = 0;
	
	foreach my $key (keys %{$smallHashTable}){
		next if (!exists $largeHashTable->{$key});
		$result++;
	}
	
	return $result;
} #END sub CardIntersection


sub CardUnion{
	#��������Hash��, ��������key��������Ŀ
	my $sHashTableOne = shift;
	my $sHashTableTwo = shift;
	
	my %unionHashTable;
	
	foreach my $key (keys %{$sHashTableOne}){
		$unionHashTable{$key} = 1;
	}

	foreach my $key (keys %{$sHashTableTwo}){
		$unionHashTable{$key} = 1;
	}
	
	my $result = 0;
	
	$result = (keys %unionHashTable);
	
	return $result;
}#END sub CardUnion


sub CAC{	
	#����:����-��������
	#����ֵ: CAC��ֵ
	my $inputMatrix = shift;	

	#������з���������������, �򷵻�0
	return 0 if (@{$inputMatrix} == 0);	
			
  my $noRow = @{$inputMatrix};
	my $noColumn = @{$inputMatrix->[0]};
	
	my $sum = 0;
	my $temp;	
	
	for (my $i = 0; $i < $noRow; $i++){			
		$temp = 0;
		for (my $j = 0; $j < $noColumn; $j++){
			$temp = $temp + $inputMatrix->[$i][$j];
		}		
		
		$sum = $sum + $temp if ($temp > 1);
	}	
	
	my $result = 0;	
	$result = $sum / ($noRow * $noColumn);	
	
	return $result;
}#END sub CAC


sub CDE{
	#����CDE
	my $class = shift;
	
	my %identifierList = ();
	
	my $sCDE = 0;
	
	my $definein = $class->ref();
	
	return 0 if (!$definein);
	
  my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($class);
  return 0 if ($lexer eq "undef");
		
	foreach my $lexeme ($lexer->lexemes($startLine, $endLine)){
		if ($lexeme->token() eq "Identifier"){
			if (!exists $identifierList{$lexeme->text()}){
				$identifierList{$lexeme->text()} = 1;
			}
			else{
				$identifierList{$lexeme->text()} = $identifierList{$lexeme->text()} + 1;
			}				
		}			
	}

  
  my %oldHash = ();
  foreach my $key (keys %identifierList){
  	$oldHash{$key} = $identifierList{$key};
  }

		
	#�����������ͷ�ļ��ж���, ��ͳ�Ʒ������еı�ʶ��			
	my @methodArray = getEntsInClass($class, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");		
		
	foreach my $func (@methodArray){
		next if (!IsMethodInClassHeader($class, $func));
		
    my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);
    next if ($lexer eq "undef");
    
    $startLine++;  #������ͷ��Ӧ��ȥ��
      
	  foreach my $lexeme ($lexer->lexemes($startLine, $endLine)) {
	    if ($lexeme->token() eq "Identifier"){
			  if ($identifierList{$lexeme->text()} <= 1){
				  delete $identifierList{$lexeme->text()};
			  }
			  else{
				  $identifierList{$lexeme->text()}--;
			  }				
		  }				  
		}
		  
	}			

#  foreach my $key (keys %oldHash){
#  	print "\n\t\t";
#  	
#  	my $temp = 0;
#  	$temp = $identifierList{$key} if (exists $identifierList{$key});
#  	
#  	print "(", $key, ",", $oldHash{$key}, "-->", $temp, ")";
#  }
		
		
	my $totalNoOfIdentifer = 0;
		
	foreach my $key (keys %identifierList){
		$totalNoOfIdentifer = $totalNoOfIdentifer + $identifierList{$key};			
	}
		
	$sCDE = 0;
		
	foreach my $key (keys %identifierList){
		$sCDE = $sCDE - $identifierList{$key}/$totalNoOfIdentifer 
		                * log($identifierList{$key}/$totalNoOfIdentifer) / log(2);			
	}		
		
		
#		print "totalNoOfIdentifer = ", $totalNoOfIdentifer, "\n";
#		foreach my $key (keys %identifierList){
#			print $key, "\t\t", $identifierList{$key}, "\n";			
#		}
			
	return $sCDE;
} # End sub CDE



sub CIE{
	#����CIE
	my $class = shift;
	
	my %identifierList = ();	
	
	my @methodArray = ();
  @methodArray = getRefsInClass($class, "define","function ~unknown ~unresolved,  method ~unknown ~unresolved");
  if (@methodArray == 0){
  	return 0; 	
  }

	
	foreach my $method (@methodArray){
		my $func = $method->ent();
		
    my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);	  
    next if ($lexer eq "undef");

	  foreach my $lexeme ($lexer->lexemes($startLine, $endLine)) {
			if ($lexeme->token() eq "Identifier"){
				if (!exists $identifierList{$lexeme->text()}){
					$identifierList{$lexeme->text()} = 1;
				}
				else{
					$identifierList{$lexeme->text()} = $identifierList{$lexeme->text()} + 1;
				}				
			}				  
		}
	}
	
	my $totalNoOfIdentifer = 0;
		
	foreach my $key (keys %identifierList){
		$totalNoOfIdentifer = $totalNoOfIdentifer + $identifierList{$key};			
	}
		
	my $sCIE = 0;
		
	foreach my $key (keys %identifierList){
		$sCIE = $sCIE - $identifierList{$key}/$totalNoOfIdentifer 
			                * log($identifierList{$key}/$totalNoOfIdentifer) / log(2);			
	}	
			  			
	return $sCIE;
} # END sub CIE


sub WMC{
	my $sClass = shift;
	
#	my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
#	my $result = 0;
#	
#	foreach my $func (@methodArray){	
#	  $result = $result + $func->metric("Cyclomatic");
#  }

  my $result = $sClass->metric("SumCyclomatic");
	
	return $result;
}#END sub WMC


sub SDMC{
	my $sClass = shift;
	
	my @CCfunc;
	
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
  my $CCAvg = 0; 
	my $sum = 0;
	
	return 0 if (@methodArray<1);
	
	foreach my $func (@methodArray){	
		my $temp = $func->metric("Cyclomatic");
		push @CCfunc, $temp;
		$CCAvg = $CCAvg + $temp;
  }
  
  $CCAvg = $CCAvg / @CCfunc;  
  
  foreach my $value (@CCfunc){
  	$sum = $sum + ($value - $CCAvg) * ($value - $CCAvg);
  }
  
  my $result = sqrt($sum / @CCfunc);
	
	
	return $result;
}


sub AvgWMC{  
	#the average of cyclomatic complexity of all methods in a class, i.e. CCAvg
	my $sClass = shift;
	
  my $result = 0;  #$sClass->metric("AvgCyclomatic"); 
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
  
  return 0 if (@methodArray<1);
  
	my $sum = 0;
	foreach my $func (@methodArray){	
		$sum = $sum + $func->metric("Cyclomatic");
  }
  
  $result = $sum / @methodArray;  
  
  
	return $result;
}


sub CCMax{
	my $sClass = shift;
	
#  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	

  my $result = $sClass->metric("MaxCyclomatic"); 
	
#	foreach my $func (@methodArray){	
#		my $temp = $func->metric("Cyclomatic");
#		next if $result >= $temp;
#		$result = $temp;
#  }
  
	return $result;	
}


sub NTM{ #number of trivial methods (its CC	 = 1)
	my $sClass = shift;
	
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
  my $result = 0; 
	
	foreach my $func (@methodArray){	
		my $temp = $func->metric("Cyclomatic");
		next if $temp > 1;
		$result = $result + 1;
  }

	return $result;		
}

sub SLOCExe{
	my $sClass = shift;
	my $result = $sClass->metric("CountLineCodeExe");
	return $result;
}

sub AvgSLOCExePerMethod{
	my $sClass = shift;
	
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
  
  return 0 if (@methodArray < 1);
  
  my $result = 0; 
  my $sum = 0;
	
	foreach my $func (@methodArray){	
		$sum = $sum + $func->metric("CountLineCodeExe");
#		print "\t func = ", $func->name(), "\t SLOCExe = ", $func->metric("CountLineCodeExe"), "\t StmtExe = ", $func->metric("CountStmtExe"), "\n";
  }
  
  $result = $sum / @methodArray;  
  	
	return $result;	
}



sub AvgSLOCPerMethod{
	my $sClass = shift;
  
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");	
  
  return 0 if (@methodArray < 1);
  
  my $result = 0; 
  my $sum = 0;
	
	foreach my $func (@methodArray){	
		$sum = $sum + $func->metric("CountLineCode");
#		print "\t func = ", $func->name(), "\t SLOC = ", $func->metric("CountLineCode"), "\n";
  }
  
  $result = $sum / @methodArray;  
  
	return $result;	
}


sub NCM{ #number of class methods declared in a method
	my $sClass = shift;
	my $result = $sClass->metric("CountDeclClassMethod");
	
	return $result;
}


sub NIM{ #number of Instance methods declared in a method
	my $sClass = shift;
	my $result = $sClass->metric("CountDeclInstanceMethod");
	
	return $result;
}


sub NLM{ #number of local methods declared in a method
	my $sClass = shift;
	my $result = $sClass->metric("CountDeclMethod");
	
	return $result;
}





sub CComplexitySerires{
	my $sClass = shift;
	my $sAncestorHash = shift;
	my $sAncestorLevel = shift;
	
	my %totalMethods = (); #������е����з���: �̳з�overriding�� + overriding + �����ӵ�
	my %totalAttributes = (); #������е���������: �̳е� + �ֲ������
	
	#�õ�ǰ��ķ��������Խ��г�ʼ��
	my @methodArray = getEntsInClass($sClass, "define", "function ~private ~unknown ~unresolved, method ~private ~unknown ~unresolved");
	foreach my $func (@methodArray){
		my $signature = getFuncSignature($func, 1);
		$totalMethods{$signature} = $func;
	}
	
	my @attributeArray = getEntsInClass($sClass, "define", "Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");
	foreach my $attribute (@attributeArray){
		my $signature = getAttributeSignature($sClass, $attribute);
		$totalAttributes{$signature} = $attribute;
	}
	
	
	#����̳еķ���������
	foreach my $level (sort keys %{$sAncestorLevel}){
		my %ancestorHash = %{$sAncestorLevel->{$level}};
		
		foreach my $classKey (keys %ancestorHash){
			my $ancestorClass = $sAncestorHash->{$classKey};
			
			#----��Ӽ̳з�overiding�ķ���-----------
			my @ancestorMethodArray = getEntsInClass($ancestorClass, "define", "function ~private ~unknown ~unresolved, method ~private ~unknown ~unresolved");
			foreach my $func (@ancestorMethodArray){
				my $signature = getFuncSignature($func, 1);
				next if (exists $totalMethods{$signature}); #��������overriding��, ��������
				$totalMethods{$signature} = $func;
			}
			
			#----�������-----------
			my @ancestorAttributeArray = getEntsInClass($ancestorClass, "define", "Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");
			foreach my $attribute (@ancestorAttributeArray){
				my $signature = getAttributeSignature($ancestorClass, $attribute);
				$totalAttributes{$signature} = $attribute;
			}
		}		
	}
	
	
	#��ӵ�ǰ���ж����˽�з���
	my @methodArray = getEntsInClass($sClass, "define", "function private ~unknown ~unresolved, method private ~unknown ~unresolved");
	foreach my $func (@methodArray){
		my $signature = getFuncSignature($func, 1);
		$totalMethods{$signature} = $func;
	}	
	
#	print "\t total methods ==> ", scalar (keys %totalMethods), "\n";
#	foreach my $signature (keys %totalMethods){
#		print "\t\t ", $signature, "\n";
#	}
#	
#	print "\n\t total attributes ==> ", scalar (keys %totalAttributes), "\n";
#	foreach my $signature (keys %totalAttributes){
#		print "\t\t ", $signature, "\n";
#	}
#	
	
		
	#�������ָ����Զ���ֵ
	
	my ($valueCC1, $valueCC2) = ClassComplexityByLLL(\%totalMethods);
	my $valueCC3 = ClassComplexityByKSW(\%totalMethods, \%totalAttributes);
	
	return ($valueCC1, $valueCC2, $valueCC3);
}#END sub CComplexitySerires


sub ClassComplexityByLLL{
	my $sAllMethodHash = shift;
	
  return (0, 0) if ((keys %{$sAllMethodHash}) < 1);
		
	my $valueCC1 = 0;
	
	my $sumLN = 0;
	my $sumCP = 0;	
			
	foreach my $signature (keys %{$sAllMethodHash}){
		my $func = $sAllMethodHash->{$signature};

#		print "\t\t func = ", $signature, "\n";		

		my $funcLN = LengthOfMethod($func);
		my $funcCP = CPOfMethod($func);

#		print "\t\t\t LN = ",  $funcLN, "\n";
#		print "\t\t\t CP = ",  $funcCP, "\n";
				
		$valueCC1 = $valueCC1 + $funcLN * $funcCP * $funcCP;

		$sumLN = $sumLN + $funcLN;
		$sumCP = $sumCP + $funcCP;
	}
	
	my $valueCC2 = $sumLN * $sumCP * $sumCP;
	
	return ($valueCC1, $valueCC2);
}#END sub ClassComplexityByLLL


sub LengthOfMethod{
	my $sEnt = shift;
	my $sAllMethodHash = shift;
	
	my $func = $sEnt;
  my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);
	return 0 if ($lexer eq "undef");
	  
	my $result = 0;
	
  foreach my $lexeme ($lexer->lexemes($startLine,$endLine)) 
  {
     $result++ if ($lexeme->token eq "Operator" || $lexeme->token eq "Identifier");
  }
  
	return $result;	
}#END sub LengthOfMethod


sub CPOfMethod{
	my $sEnt = shift;
	
	my $func = $sEnt;
	
	#����Input coupling for this method
	my $result = InputCouplingOfMethod($func);
	
	#����Output coupling for this method
	#---�������õķ���---
	my @calledMethodArray = $func->refs("call", "function ~unknown ~unresolved, method ~unknown ~unresolved");
	foreach my $calledMethod (@calledMethodArray){		
		my $calledFunc = $calledMethod->ent();		
		$result = $result + InputCouplingOfMethod($calledFunc);
	}
	
	#---�������ʵķǾֲ�����---
	my @variableArray = $func->refs("use, set, modify", "object ~local ~unknown ~unresolved, variable ~local ~unknown ~unresolved");
	$result = $result + scalar @variableArray;
	
	return $result;
}#END sub CPOfMethod



sub InputCouplingOfMethod{
	my $sEnt = shift;
	
	my $func = $sEnt;
		
	my $result = 0;	
	my @parameterList = $func->ents("define", "parameter");
	$result = 1 + scalar @parameterList;	
	$result++ if ($func->type() and $func->type() !~ m/void/i); #����з���ֵ, ������1	
	
	return $result;	
}#END sub InputCouplingOfMethod



sub ClassComplexityByKSW{
	my $sAllMethodHash = shift;
	my $sAllAttributeHash = shift;
	
	my $methodNodeHash;  #keyΪ������, vlaueΪ{incoming => {�ڵ���=>������/����}, outgoing =>}
	my $attributeNodeHash;
	
	foreach my $signature (keys %{$sAllMethodHash}){
		my $func = $sAllMethodHash->{$signature};		
		
		#�������õķ���: <calling--->called>
		my @calledMethodArray = $func->refs("call", "function ~unknown ~unresolved, method ~unknown ~unresolved");
		foreach my $calledMethod (@calledMethodArray){
			my $calledFunc = $calledMethod->ent();
			my $calledSignature = getFuncSignature($calledFunc, 1);
			
			next if (!exists $sAllMethodHash->{$calledSignature}); #ֻ���Ǳ����еķ���������֮��Ľ���
			
			$methodNodeHash->{$signature}->{outgoing}->{$calledSignature}++;
			$methodNodeHash->{$calledSignature}->{incoming}->{$signature}++;
		}


		#������ʵ�����
		#----����"������": <����--->����>-------
		my @attributeReadArray = $func->refs("use", "Member Object ~local ~unknown ~unresolved, Member Variable ~local ~unknown ~unresolved");
		foreach my $attribute (@attributeReadArray){			
			my $attributeClass = $attribute->ent()->ref("definein", "Class ~unknown ~unresolved");
			next if (!$attributeClass);			

			my $attributeSignature = getAttributeSignature($attributeClass->ent(), $attribute->ent());
			
			next if (!exists $sAllAttributeHash->{$attributeSignature}); #ֻ���Ǳ����еķ���������֮��Ľ���
			
			$methodNodeHash->{$signature}->{incoming}->{$attributeSignature}++;
			$attributeNodeHash->{$attributeSignature}->{outgoing}->{$signature}++;
		}
		
		#----����"д����": <����--->����>-------
		my @attributeWriteArray = $func->refs("set", "Member Object ~local ~unknown ~unresolved, Member Variable ~local ~unknown ~unresolved");
		foreach my $attribute (@attributeWriteArray){			
			my $attributeClass = $attribute->ent()->ref("definein", "Class ~unknown ~unresolved");
			next if (!$attributeClass);			
						
			my $attributeSignature = getAttributeSignature($attributeClass->ent(), $attribute->ent());
			
			next if (!exists $sAllAttributeHash->{$attributeSignature}); #ֻ���Ǳ����еķ���������֮��Ľ���
			
			$methodNodeHash->{$signature}->{outgoing}->{$attributeSignature}++;
			$attributeNodeHash->{$attributeSignature}->{incoming}->{$signature}++;
		}		
		
		#----����"�޸�����": <����--->����> and <����--->����>-------
		my @attributeModifyArray = $func->refs("Modify", "Member Object ~local ~unknown ~unresolved, Member Variable ~local ~unknown ~unresolved");
		foreach my $attribute (@attributeModifyArray){			
			my $attributeClass = $attribute->ent()->ref("definein", "Class ~unknown ~unresolved");
			next if (!$attributeClass);			
						
			my $attributeSignature = getAttributeSignature($attributeClass->ent(), $attribute->ent());
			
			next if (!exists $sAllAttributeHash->{$attributeSignature}); #ֻ���Ǳ����еķ���������֮��Ľ���

			$methodNodeHash->{$signature}->{incoming}->{$attributeSignature}++;
			$methodNodeHash->{$signature}->{outgoing}->{$attributeSignature}++;
			$attributeNodeHash->{$attributeSignature}->{incoming}->{$signature}++;
			$attributeNodeHash->{$attributeSignature}->{outgoing}->{$signature}++;
		}		
	}
	
	
#	print "method Node Hash ===> \n";
#	
#	foreach my $signature (keys %{$methodNodeHash}){
#		print "\t ", $signature, "\n";
#		print "\t\t incoming:\n";
#		if (exists $methodNodeHash->{$signature}->{incoming}){
#			my %incoming = %{$methodNodeHash->{$signature}->{incoming}}; 
#			foreach my $key (keys %incoming){
#				print "\t\t\t ", $key, ",", $incoming{$key}, "\n"; 
#			}
#		}
#
#		print "\t\t outgoing:\n";
#		if (exists $methodNodeHash->{$signature}->{outgoing}){
#			my %outgoing = %{$methodNodeHash->{$signature}->{outgoing}}; 
#			foreach my $key (keys %outgoing){
#				print "\t\t\t", $key, ",", $outgoing{$key}, "\n"; 
#			}
#		}	
#	}
#	
#	
#	print "\n attribute Node Hash ===> \n";
#	
#	foreach my $signature (keys %{$attributeNodeHash}){
#		print "\t ", $signature, "\n";
#		print "\t\t incoming:\n";
#		if (exists $attributeNodeHash->{$signature}->{incoming}){
#			my %incoming = %{$attributeNodeHash->{$signature}->{incoming}}; 
#			foreach my $key (keys %incoming){
#				print "\t\t\t", $key,",", $incoming{$key}, "\n"; 
#			}
#		}
#
#		print "\t\t outgoing:\n";
#		if (exists $attributeNodeHash->{$signature}->{outgoing}){
#			my %outgoing = %{$attributeNodeHash->{$signature}->{outgoing}}; 
#			foreach my $key (keys %outgoing){
#				print "\t\t\t", $key, ",", $outgoing{$key}, "\n"; 
#			}
#		}	
#	}	
	
		
	
	my @probabilityArray; #��¼ÿ���ڵ�ĸ���
	my $sum = 0;   #������ֵ�����ܱ�����2��
	
	foreach my $signature (keys %{$methodNodeHash}){
		my $temp = 0;
		
		if (exists $methodNodeHash->{$signature}->{incoming}){
			my %incoming = %{$methodNodeHash->{$signature}->{incoming}};  		
	  	foreach my $key (keys %incoming){
		  	$temp = $temp + $incoming{$key};
		  }
		}
		
		if (exists $methodNodeHash->{$signature}->{outgoing}){	   
			my %outgoing = %{$methodNodeHash->{$signature}->{outgoing}};
			
		  foreach my $key (keys %outgoing){			  
		  	$temp = $temp + $outgoing{$key};
		  }
		}
		
		$sum = $sum + $temp;		
		push @probabilityArray, $temp;
	}
	
	foreach my $signature (keys %{$attributeNodeHash}){
		my $temp = 0;
		
		if (exists $attributeNodeHash->{$signature}->{incoming}){
		  my %incoming = %{$attributeNodeHash->{$signature}->{incoming}};
  		foreach my $key (keys %incoming){
	   		$temp = $temp + $incoming{$key};
		  }
		}
		  
		if (exists $attributeNodeHash->{$signature}->{outgoing}){  
		  my %outgoing = %{$attributeNodeHash->{$signature}->{outgoing}};
  		foreach my $key (keys %outgoing){
	  		$temp = $temp + $outgoing{$key};
		  }
		}
		
		$sum = $sum + $temp;		
		push @probabilityArray, $temp;
	}	
	
	return 0 if ($sum == 0);
	
	my $valueCC3 = 0;
	
	for (my $i = 0; $i < @probabilityArray; $i++){
		$probabilityArray[$i] = $probabilityArray[$i] / $sum;
		$valueCC3 = $valueCC3 - log($probabilityArray[$i]) * $probabilityArray[$i] / log(2);
	}	
	
	return $valueCC3;
}#END sub ClassComplexityByKSW


sub NMIMP{
	my $sClass = shift;
	
#	my $result = $sClass->metric("CountDeclMethod");
  my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved"); 
  my $result = scalar @methodArray;	
  
	# print "\t\t NMIMP = ", $result, "\n";
	
	return $result;
}#END sub NMIMP


sub NAIMP{
	my $sClass = shift;
	
#	my $result = $sClass->metric("CountDeclClassVariable") + $sClass->metric("CountDeclInstanceVariable");

  my @attributeArray = getEntsInClass($sClass, "define", "Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");  
  my $result = scalar @attributeArray;	
	# print "\t\t NAIMP = ", $result, "\n";
	
	my $dit = $sClass->metric("MaxInheritanceTree");
	# print "\t\t DIT = ", $dit, "\n";
	
	return $result;
}



sub UnderstandSLOC{
	my $sClass = shift;
	
	return $sClass->metric("CountLineCode");
}


sub C3{
	my $class = shift;
	
	my %Vocabulary = ();	
	
	my @methodArray = ();
  @methodArray = getRefsInClass($class, "define","function ~unknown ~unresolved,  method ~unknown ~unresolved");
  if (@methodArray < 2){
  	return wantarray?(1,0):1; 	
  }
  
	foreach my $method (@methodArray){
		my $func = $method->ent();
		
    my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);
    next if ($lexer eq "undef");

	  foreach my $lexeme ($lexer->lexemes($startLine, $endLine)) {
			if ($lexeme->token() eq "Identifier"){				
					$Vocabulary{$lexeme->text()}->{$func->id()} = 1;												
			}				  
		}		
	}
	
	my %noForTerm = ();	
	my $jj = 0;
	
	foreach my $termKey (keys %Vocabulary) {
		$noForTerm{$termKey} = $jj;
		$jj++;		
	}
	
	my $lengthOfVector = scalar (keys %Vocabulary);
	
  if ($lengthOfVector < 1){
  	my $temp = @methodArray;
  	#�κ������������֮�䶼�����ڱ�
  	my $specialLCSM = $temp*($temp - 1)/2;  	
  	return wantarray?(0,$specialLCSM):0; 	
  }	
	
	
	my %termVectorForMethods = ();
	
	foreach my $method (@methodArray){
		my $func = $method->ent();
		
    my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);	  
    next if ($lexer eq "undef");
    
	  my %termFrequency = ();
	  my $totalTerm = 0; 
	  
	  foreach my $lexeme ($lexer->lexemes($startLine, $endLine)) {
			next if ($lexeme->token() ne "Identifier");
			$totalTerm++;
			if (!exists $termFrequency{$lexeme->text()}){
				$termFrequency{$lexeme->text()} = 1;
			}				
			else{
				$termFrequency{$lexeme->text()} = $termFrequency{$lexeme->text()} + 1;				
			}				
		}		
		
		foreach my $termKey (keys %termFrequency){
			$termFrequency{$termKey} = $termFrequency{$termKey} / $totalTerm;			
		}		
		
		for (my $i = 0; $i < $lengthOfVector; $i++){
			$termVectorForMethods{$func->id()}->[$i] = 0;
		}
		

		foreach my $termKey (keys %termFrequency){
			my $noDocuments = scalar (keys %{$Vocabulary{$termKey}});		
			$termVectorForMethods{$func->id()}->[$noForTerm{$termKey}] = 
			   $termFrequency{$termKey} * log(@methodArray / $noDocuments) / log(2);			
		}		
	}
	
	my $noMethod = @methodArray;
	
	my @similarityMatrix = ();
	
	for (my $i = 0; $i < $noMethod; $i++){
		for (my $j = 0; $j < $noMethod; $j++){			
			$similarityMatrix[$i][$j] = 0;			
			$similarityMatrix[$j][$i] = 0;			
		}		
	}
	
	
	for (my $i = 0; $i < $noMethod - 1; $i++){
		my $firstMethod = $methodArray[$i]->ent()->id();		
		for (my $j = $i + 1; $j < $noMethod; $j++){
			my $secondMethod = $methodArray[$j]->ent()->id();					
			$similarityMatrix[$i][$j] = 
			      vectorSimilarity($termVectorForMethods{$firstMethod}, $termVectorForMethods{$secondMethod});			
			$similarityMatrix[$j][$i] = $similarityMatrix[$i][$j];
		}		
	}
  
  
  
  my $ACSM = 0;  
  my $sum = 0;
  
  for (my $i = 0; $i < $noMethod; $i++){
  	for (my $j = 0; $j < $noMethod; $j++){
  		$sum = $sum + $similarityMatrix[$i][$j];  		
  	}  	
  }
  
  $ACSM = $sum / ($noMethod * ($noMethod - 1));  
  
  my $sC3 = 0;
  
  $sC3 = $ACSM if ($ACSM > 0);

  my @tempMethodMethodMatrix = ();
  
  for (my $i = 0; $i < $noMethod; $i++){
  	for (my $j = $i; $j < $noMethod; $j++){  		
  		if ($similarityMatrix[$i][$j] > $ACSM){
  			$tempMethodMethodMatrix[$i][$j] = 1;  		
  			$tempMethodMethodMatrix[$j][$i] = 1;
  		}
  		else{
  			$tempMethodMethodMatrix[$i][$j] = 0;  		
  			$tempMethodMethodMatrix[$j][$i] = 0;
  		}
  	}  	
  }

  
  my @intersectionMatrix = ();
  
  for (my $i = 0; $i < $noMethod; $i++){
  	for (my $j = 0; $j < $noMethod; $j++){
  		$intersectionMatrix[$i][$j] = 0;
  	}
  }
  
  for (my $i = 0; $i < $noMethod - 1; $i++){
  	for (my $j = $i + 1; $j < $noMethod; $j++){
  		my @arrayOne = @{$tempMethodMethodMatrix[$i]};
  		my @arrayTwo = @{$tempMethodMethodMatrix[$j]};
  		
  		if (isMethodSimilar(\@arrayOne, \@arrayTwo)){
  			$intersectionMatrix[$i][$j] = 1;
  			$intersectionMatrix[$j][$i] = 1;
  		}  		
  	}
  }
  
  my $sLCSM = LCOM2(\@intersectionMatrix);
  
  return wantarray?($sC3, $sLCSM): $sC3;	
} # End sub C3


sub isMethodSimilar{
	my $arrayOne = shift;
	my $arrayTwo = shift;
	
	my $similar = 0;
	
	my $i = 0;
	
	while (!$similar && $i < @{$arrayOne}){
		$similar = 1 if (($arrayOne->[$i] == 1) && ($arrayTwo->[$i] == 1));
		$i++;
	}
	
	return $similar;	
} # END sub isMethodSimilar


sub vectorSimilarity{
	my $firstVector = shift;
	my $secondVector = shift;
	
	my $sum0 = 0;
	my $sum1 = 0;
	my $sum2 = 0;
	
	
	my $noElem = @{$firstVector};	
	
	for (my $i = 0; $i < $noElem; $i++){
		$sum0 = $sum0 + $firstVector->[$i] * $secondVector->[$i];		
		$sum1 = $sum1 + $firstVector->[$i] * $firstVector->[$i];		
		$sum2 = $sum2 + $secondVector->[$i] * $secondVector->[$i];						
	}
	
  return "Undefined" if ($sum1 == 0 || $sum2 == 0);
  
  my $result = $sum0 / (sqrt($sum1)*sqrt($sum2));
  
  return $result;	
} # END sub vectorSimilarity



sub LCOM1{ 
   #----����LCOM1-----
   my $sMethodMethodMatrix = shift;
   
   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 0 if (@{$sMethodMethodMatrix} == 1);
   
   my $noRowOrCol = @{$sMethodMethodMatrix};   
   my $sLCOM1 = 0;
   
   for (my $i = 0; $i < $noRowOrCol; $i++){
   	for (my $j = 0; $j < $noRowOrCol; $j++){
   		$sLCOM1++ if (($i != $j) && ($sMethodMethodMatrix->[$i][$j] == 0));
   	}
   }    
  
   $sLCOM1 = $sLCOM1 / 2;
    
   return $sLCOM1;
} #END sub LCOM1
  

sub LCOM2{   
   #----����LCOM2-----
   my $sMethodMethodMatrix = shift;

   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 0 if (@{$sMethodMethodMatrix} == 1); 
  
   my $noRowOrCol = @{$sMethodMethodMatrix};
   my $sLCOM2 = 0;
   my $noSimilar = 0;
   my $noNonSimilar = 0;
   
   for (my $i = 0; $i < $noRowOrCol; $i++){
   	for (my $j = 0; $j < $noRowOrCol; $j++){
   		next if ($i == $j);
   		if ($sMethodMethodMatrix->[$i][$j] == 0){
   			$noNonSimilar++;
   		}
   		else
   		{
   			$noSimilar++;
   		}   		
   	}
   }   
   
   $noNonSimilar = $noNonSimilar / 2;
   $noSimilar = $noSimilar / 2;
   
   $sLCOM2 = $noNonSimilar - $noSimilar if ($noNonSimilar - $noSimilar > 0);

   return $sLCOM2;
 } #END sub LCOM2
 
   
sub LCOM3{
   #----����LCOM3-----
   my $sMethodMethodMatrix = shift;

   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 1 if (@{$sMethodMethodMatrix} == 1);

   my $noRowOrCol = @{$sMethodMethodMatrix};
   my $sLCOM3 = 0;
   my @visited;
   
   for (my $i = 0; $i < $noRowOrCol; $i++){
    	$visited[$i] = 0;	
   }
   
   for (my $i = 0; $i < $noRowOrCol; $i++){ 
   	if (!$visited[$i]){  
   		$sLCOM3++;
   		depthFirstSearch($sMethodMethodMatrix, $i,\@visited); 
    }  	
   }
   
   return $sLCOM3;
} # END sub LCOM3
   

sub Co{
   #----����Co-----
   my $sMethodMethodMatrix = shift;   

   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 1 if (@{$sMethodMethodMatrix} == 1);
  
   my $noRowOrCol = @{$sMethodMethodMatrix};     
   my $noEdge = 0;
   my $noVetex = $noRowOrCol;   
   my $sCo = 0;
      
   for (my $i = 0; $i < $noRowOrCol; $i++){
   	for (my $j = 0; $j < $noRowOrCol; $j++){
   		next if ($i == $j);
      $noEdge++ if ($sMethodMethodMatrix->[$i][$j] == 1);
    }
   }
   
   $noEdge = $noEdge / 2; 
   
   if ($noVetex == 2){
   	 return 0 if ($noEdge == 0);
     return 1;
   }   
  
   $sCo = 2 * ($noEdge - $noVetex + 1) / (($noVetex - 1) * ($noVetex - 2));
      
   return $sCo;
} # END sub Co


sub NewCo{
   #----����NewCo-----
   my $sMethodMethodMatrix = shift;   

   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 1 if (@{$sMethodMethodMatrix} == 1);
   
  
   my $noRowOrCol = @{$sMethodMethodMatrix};     
   my $noEdge = 0;
   my $noVetex = $noRowOrCol;   
   my $sNewCo;
      
   for (my $i = 0; $i < $noRowOrCol; $i++){
   	for (my $j = 0; $j < $noRowOrCol; $j++){
   		next if ($i == $j);
      $noEdge++ if ($sMethodMethodMatrix->[$i][$j] == 1);
    }
   }
   
   $noEdge = $noEdge / 2; 
  
   $sNewCo = 2 * $noEdge / ($noVetex * ($noVetex - 1));
   
   return $sNewCo;
} #END sub NewCo
 

sub LCOM5{
   #----����LCOM5-----
   my $sAttributeMethodMatrix = shift;

   return "undefined" if (@{$sAttributeMethodMatrix} == 0);
   
   my $sLCOM5 = 0;

   my $noRow = @{$sAttributeMethodMatrix};
   my $noCol = @{$sAttributeMethodMatrix->[0]};

   return "undefined" if ($noCol == 0);
   return 0 if ($noCol == 1);
   
   my $sum = 0;   
      
   for (my $i = 0; $i < $noRow; $i++){
   	for (my $j = 0; $j < $noCol; $j++){
      $sum = $sum + 1 if ($sAttributeMethodMatrix->[$i][$j] == 1);
    }
   }
 
   $sLCOM5 = ($noCol - $sum / $noRow) / ($noCol-1);

   return $sLCOM5;
} # END sub LCOM5


sub NewLCOM5{
   #----����NewCoh, Briand�����LCOM5����-----
   my $sAttributeMethodMatrix = shift;

   if ((@{$sAttributeMethodMatrix} == 0) || (@{$sAttributeMethodMatrix->[0]}==0)) {
    	return "undefined";
   }   
    
   my $sNewCoh = 0;

   my $noRow = @{$sAttributeMethodMatrix};
   my $noCol = @{$sAttributeMethodMatrix->[0]};

   my $sum = 0;
   
      
   for (my $i = 0; $i < $noRow; $i++){
   	for (my $j = 0; $j < $noCol; $j++){
      $sum = $sum + 1 if ($sAttributeMethodMatrix->[$i][$j] == 1);
    }
   }
 
   
  $sNewCoh = ($sum / ($noRow * $noCol));
    
  return $sNewCoh;
}	# END sub NewLCOM5


sub LCOM6{
   #----����LCOM6-----
   my $sClass = shift;
   
   my @methodArray = getEntsInClass($sClass, "define", "function ~unknown, method ~unknown");
   
   return "undefined" if (@methodArray == 0);
      
   my %parameterNameHash; #keyΪ������, valueΪ������. ��ʾ��Щ�������иò���. ʵ��ֻ��¼һ��������
   my %methodNameHash; #keyΪ������, valueΪ�������. ��ʾ�÷��������ĸ�����
   
   my $currentSetNo = 0;  #��ǰ�ļ��Ϻ�, ��ֵΪ0. ���ϴ�1��ʼ���
   
   foreach my $func (@methodArray){
   	 my @parameterList = $func->ents("define", "parameter");
   	 
   	 my $hasCommonPara; # ����ǰɨ��ķ����й�������?
   	 $hasCommonPara = 0;
   	 foreach my $parameter (@parameterList){
   	 	 if (exists $parameterNameHash{$parameter->name()}){
   	 	 	 $hasCommonPara = 1;    	 	 
   	 	 	 my $previousMethodName = $parameterNameHash{$parameter->name()};
   	 	   $methodNameHash{getFuncSignature($func, 1)} = $methodNameHash{$previousMethodName};
   	 	 }
   	 	 else{
   	 	 	 $parameterNameHash{$parameter->name()} = getFuncSignature($func,1);
   	 	 }
   	 }
   	 
   	 if (!$hasCommonPara){
   	 	 $currentSetNo++;
   	 	 $methodNameHash{getFuncSignature($func,1)} = $currentSetNo;
   	 }
   }
   
   my $result = 0;
   foreach my $key (%methodNameHash){
   	$result = $methodNameHash{$key} if $result < $methodNameHash{$key};
   }
   
   $result = 100 * $result / @methodArray;
  
   return $result;	
}#END sub LCOM6


sub TCC{
   #----����TCC-----
   my $sMethodMethodMatrix = shift;
   
   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 1 if (@{$sMethodMethodMatrix} == 1);
   
   my $noRowOrCol = @{$sMethodMethodMatrix};
   my $NDC = 0;
   my $NP = $noRowOrCol * ($noRowOrCol - 1)/2; 
   my $sTCC = 0;

   for (my $i = 0; $i < $noRowOrCol; $i++){
   	for (my $j = 0; $j < $noRowOrCol; $j++){
   		next if ($i == $j);
   		$NDC++ if ($sMethodMethodMatrix->[$i][$j] == 1);  	
   	}
   }
   $NDC = $NDC / 2;
  
   if ($NP > 0){
   	$sTCC = $NDC / $NP;
   }

   return $sTCC;
} #END sub TCC


sub LCC{
   #----����LCC-----
   my $sMethodMethodMatrix = shift;
   
   return "undefined" if (@{$sMethodMethodMatrix} == 0);
   return 1 if (@{$sMethodMethodMatrix} == 1);    
   
   
   my $noRowOrCol = @{$sMethodMethodMatrix};
   my $NIC = 0;
   my $NP = $noRowOrCol * ($noRowOrCol - 1)/2;      
   
   my $sLCC = 0;
   
   my @visited;
   my @noElemOfsubG = (); #ÿ����ͨ��ͼ�а����Ľ�����
   
   for (my $i = 0; $i < $noRowOrCol; $i++){
    	$visited[$i] = 0;	
   }
   
   for (my $i = 0; $i < $noRowOrCol; $i++){ 
   	if (!$visited[$i]){     		
   		my $before = 0;
   		for (my $j = 0; $j < $noRowOrCol; $j++){
   			$before++ if ($visited[$j] == 1);
   		}
   		depthFirstSearch($sMethodMethodMatrix, $i,\@visited); 
   		my $after = 0;
   		for (my $j = 0; $j < $noRowOrCol; $j++){
   			$after++ if ($visited[$j] == 1);
   		}
   		push @noElemOfsubG, $after - $before;   		
    }  	
   }

   for (my $i = 0; $i < @noElemOfsubG; $i++){
     $NIC = $NIC + $noElemOfsubG[$i]*($noElemOfsubG[$i] - 1) / 2;
   }
   
   if ($NP > 0){
   	$sLCC = $NIC / $NP;   
   }

   return $sLCC;
} # END sub LCC
 
 
 
sub OCC{
   #----����OCC-----
   my $sMethodMethodMatrix = shift;	 
   
   #��¼��ÿ���������������ɵ���ķ�����   
  
   my $noElem = @{$sMethodMethodMatrix};
   
   return "Undefined" if ($noElem == 0 );   
   return 0 if ($noElem == 1);
   
   my @NoOfRechableMethods = ();
   my @visited;  
   my $count = 0;
   
 
   for (my $node = 0; $node < $noElem; $node++){

   	  for (my $j = 0; $j < $noElem; $j++){
   		  $visited[$j] = 0;	
   	  }    	
   	
    	depthFirstSearch($sMethodMethodMatrix, $node,\@visited);     	    	
    	
    	$count = 0;
 	   	for (my $j = 0; $j < $noElem; $j++){
   		   $count++ if ($visited[$j] == 1);
   	  }
   	
   	  $NoOfRechableMethods[$node] = $count - 1;   	   	  
   }
   
   
   my $max = 0;
   
   for (my $node = 0; $node < $noElem; $node++){
   	 $max = $NoOfRechableMethods[$node] if ($NoOfRechableMethods[$node] > $max);   
   }
   
   my $sOCC = $max / ($noElem - 1);
      
   return $sOCC; 
} # END sub OCC


sub PCC{
	my $sAttributeReadTable = @_[0];
	my $sAttributeWriteTable = @_[1];
	my $sAttributeModifyTable = @_[2];
	my $sMethodWithoutAttributeParaTable = @_[3];	
	
	
#	print "noOfattributeRead = ", scalar (keys %{$sAttributeReadTable}), "\n";
#	print "noOfattributeWrite = ", scalar (keys %{$sAttributeWriteTable}), "\n";
#	print "noOfattributeModify = ", scalar (keys %{$sAttributeModifyTable}), "\n";
	
	my %allMethodList = ();
	
  foreach my $aHashRef (@_){   	 
  	foreach my $attributeKey (sort keys %{$aHashRef}){
  		my %tempMethodHashTable = %{$aHashRef->{$attributeKey}};
  		foreach my $methodKey (sort keys %tempMethodHashTable){
  			$allMethodList{$methodKey} = 1;
  		}
  	}
  }
  
  my $noElem = scalar (keys %allMethodList);
  
  return "Undefined" if ($noElem == 0);  
  return 0 if ($noElem == 1);
  
  my $jj=0;
  foreach my $methodKey (sort keys %allMethodList){
   	$allMethodList{$methodKey} = $jj;
   	$jj++;  	
  } 

  
  my @tempMethodMethodMatrix = ();
  
    
  for (my $i = 0; $i < $noElem; $i++){
  	for (my $j = 0; $j < $noElem; $j++){
  		$tempMethodMethodMatrix[$i][$j] = 0;
  	} 	
  }
  
  #����"����д�Ͷ��б�", ����������������ϵ
  
  foreach my $attributeKey (sort keys %{$sAttributeWriteTable}){
  	next if (!exists $sAttributeReadTable->{$attributeKey});
  	
  	my %tempHashTable = $sAttributeWriteTable->{$attributeKey};
  	
  	foreach my $fstMethodKey (keys %{$sAttributeWriteTable->{$attributeKey}}){
  		foreach my $sndMethodKey (keys %{$sAttributeReadTable->{$attributeKey}}){  			  			
  			my $row = $allMethodList{$fstMethodKey};
  			my $col = $allMethodList{$sndMethodKey};  			
  			$tempMethodMethodMatrix[$row][$col] = 1;  			
  		}  		
  	} 	
  } 
  
  #����"����д���޸��б�", ����������������ϵ  
  foreach my $attributeKey (sort keys %{$sAttributeWriteTable}){
  	next if (!exists $sAttributeModifyTable->{$attributeKey});
  	
  	foreach my $fstMethodKey (keys %{$sAttributeWriteTable->{$attributeKey}}){
  		foreach my $sndMethodKey (keys %{$sAttributeModifyTable->{$attributeKey}}){  			
  			my $row = $allMethodList{fstMethodKey};
  			my $col = $allMethodList{sndMethodKey};  			
  			$tempMethodMethodMatrix[$row][$col] = 1;  			
  		}  		
  	} 	
  } 
 
  #����"�����޸ĺͶ��б�", ����������������ϵ  
  foreach my $attributeKey (sort keys %{$sAttributeModifyTable}){
  	next if (!exists $sAttributeReadTable->{$attributeKey});
  	
  	foreach my $fstMethodKey (keys %{$sAttributeModifyTable->{$attributeKey}}){
  		foreach my $sndMethodKey (keys %{$sAttributeReadTable->{$attributeKey}}){  			
  			my $row = $allMethodList{fstMethodKey};
  			my $col = $allMethodList{sndMethodKey};  			
  			$tempMethodMethodMatrix[$row][$col] = 1;  			
  		}  		
  	} 	
  } 
  
  my $sPCC = OCC(\@tempMethodMethodMatrix);
  
  return $sPCC; 
} # END sub PCC
 
 
 


#####����CBMC�ĺ���#########

sub CBMC{
	my $sAttributeMethodMatrix = shift;	
		
	return 1 if (@{$sAttributeMethodMatrix} == 0); #ֻ��һ�����Խڵ���߷����ڵ��ͼ
	
	return 1 if (isMCC($sAttributeMethodMatrix));
	return 0 if (isDisjoint($sAttributeMethodMatrix));
	
	my @arrayOfGlueMethodSet = getArrayOfGlueMethodSet($sAttributeMethodMatrix);

  my $maxCBMC = 0;

	my $Fs = 0;
	my $Fc = 0;
	
  foreach my $currentGlueMethodSet (@arrayOfGlueMethodSet){    
  	
  	my $noSubGraphs = 0;
  	my @methodArray = ();
  	my @attributeArray = ();
  	
  	my @graphWithoutGlueMethods = excludeGlueMethods($sAttributeMethodMatrix, $currentGlueMethodSet);  	
  	
  	$noSubGraphs = getNumberOfSubGraphs(\@graphWithoutGlueMethods, \@methodArray, \@attributeArray);
  	
  	my $sum;
  	
  	$sum = 0;
  	
  	for (my $i = 1; $i <= $noSubGraphs; $i++){
  		my @aSubGraph = getSubGraph(\@graphWithoutGlueMethods, $i, \@methodArray, \@attributeArray);  		
  		my $temp = CBMC(\@aSubGraph); 		
  		$sum = $sum + $temp;
  	}
  	
  	$Fs = $sum / $noSubGraphs;  	
  	$Fc = @{$currentGlueMethodSet} / @{$sAttributeMethodMatrix->[0]};  	
 	  	
  	$maxCBMC = $Fs * $Fc if ($Fs * $Fc > $maxCBMC);
  }  
  
  return $maxCBMC;	
} # END sub CBMC


sub excludeGlueMethods{
	my $sAttributeMethodMatrix = shift;
	my $sCurrentGlueMethodSet = shift;
	
	my @resultGraph = ();
	
	my @attributeArray = ();
	my @methodArray = ();
	
	for (my $attribute = 0; $attribute < @{$sAttributeMethodMatrix}; $attribute++){
		$attributeArray[$attribute] = $attribute;
	}
	
	for (my $method = 0; $method < @{$sAttributeMethodMatrix->[0]}; $method++){
		next if (findElem($sCurrentGlueMethodSet, $method));
		push @methodArray, $method;		
	}
	
	for (my $row = 0; $row < @attributeArray; $row++){
		for (my $col = 0; $col < @methodArray; $col++){
			$resultGraph[$row][$col] = $sAttributeMethodMatrix->[$attributeArray[$row]][$methodArray[$col]];			
		}		
	}	
		
	return @resultGraph;
} # End Sub excludeGlueMethods




sub getSubGraph{
	#���ص�k����ͼ
	my $sAttributeMethodMatrix = shift;
	my $kthSubGraph = shift;
	my $sMethodInGraph = shift;
	my $sAttributeInGraph = shift;
	
	my @aSubGraph = ();
	my @subGraphAttribute = ();
	my @subGraphMethod = ();
	
	for (my $attribute = 0; $attribute < @{$sAttributeInGraph}; $attribute++){
		next if ($sAttributeInGraph->[$attribute] != $kthSubGraph);
		push @subGraphAttribute, $attribute;		
	}
	
	for (my $method = 0; $method < @{$sMethodInGraph}; $method++){
		next if ($sMethodInGraph->[$method] != $kthSubGraph);
		push @subGraphMethod, $method;		
	}
	
	for (my $row = 0; $row < @subGraphAttribute; $row++){
		for (my $col = 0; $col < @subGraphMethod; $col++){
			$aSubGraph[$row][$col] = $sAttributeMethodMatrix->[$subGraphAttribute[$row]][$subGraphMethod[$col]];			
		}		
	}
	
	return @aSubGraph; 
} # END sub getSubGraph



sub isMCC{
#�ж�һ��ͼ�Ƿ���"MCC"(�κ�һ���������������е�����)
#Pre: �����ͼ����ͨͼ,��������һ�����Խڵ�ͷ����ڵ�
   my $sAttributeMethodMatrix = shift;
   
   my $isMCC = 0;
   
   my $noRow = @{$sAttributeMethodMatrix};      
   my $noCol = @{$sAttributeMethodMatrix->[0]};
   
   my $i = 0;
   my $j = 0;
   
   my $findZero = 0;
   
  
   while ($i < $noRow && !$findZero){   	
   	$j = 0;
   	while ($j < $noCol && !$findZero){
  		$findZero = 1 if ($sAttributeMethodMatrix->[$i][$j] == 0); 
   		$j++;   		
   	}
   	$i++;  
   }
  
   $isMCC = 1 if (!$findZero);   	
   	
	 return $isMCC;
} # END sub isMCC
 

sub getArrayOfGlueMethodSet{
	#����:һ����ͨͼ
	#���: "��ˮ"����������, ���ݽṹ: 2ά����, ÿ�ж�Ӧ��һ����ˮ������ (һ����ͨͼ�����ж����ˮ������)
	my $sAttributeMethodMatrix = shift;	 
	my @arrayOfGlueMethodSet = ();
		
	my $found = 0;	#�������һ����ˮ������, �����˳�. ��Ϊ�����彺ˮ������ʹ����ͨͼ��Ϊ����ͨͼ����С������
	my $currentNoElemInGlueMethodSet = 1;  #��ǰ�Ľ�ˮ�������а�����Ԫ����Ŀ
  my $maxNoElemInGlueMethodSet = @{$sAttributeMethodMatrix->[0]}; #��ˮ��������Ԫ����Ŀ�����ֵ	
    
	while (!$found && $currentNoElemInGlueMethodSet <= $maxNoElemInGlueMethodSet){
		 my $maxNoElem = $maxNoElemInGlueMethodSet - 1; #����Ԫ�ر��, ��Ϊ����Ǵ�0��ʼ
		 
		 my @allArrayOfMethodSet = ();
		 my @tempArray = ();
		 
     getSelectedSet(\@allArrayOfMethodSet, \@tempArray, 0, 0, $currentNoElemInGlueMethodSet, $maxNoElem);		
     
     foreach my $currentSet (@allArrayOfMethodSet){     
     	  if (isAGlueMethodSet($sAttributeMethodMatrix, $currentSet)){
     		   my $row = @arrayOfGlueMethodSet;
     		   for (my $col = 0; $col < @{$currentSet}; $col++){
     			    $arrayOfGlueMethodSet[$row][$col] = $currentSet->[$col];     			
     		   }#end for     		
     	  } #end if         	
     }#end for
     
     $found = 1 if (@arrayOfGlueMethodSet);
     $currentNoElemInGlueMethodSet++;	
	}
		
	return @arrayOfGlueMethodSet;	
} # END sub getArrayOfGlueMethodSet
 

sub isAGlueMethodSet{
	my $sAttributeMethodMatrix = shift;	 
	my $sCurrentSet = shift;	
		
	my @tempAttributeMethodMatrix = ();
	
	my $row = @{$sAttributeMethodMatrix};
	my $col = @{$sAttributeMethodMatrix->[0]};
	
	return 1 if ($col == @{$sCurrentSet}); #�����ǰ�������а������еķ���, ��ô��һ���ǽ�ˮ������ (������Ŀ����1�������)
		
	for (my $i = 0; $i < $row; $i++){		
		for (my $j = 0; $j < $col; $j++){
			next if (findElem($sCurrentSet, $j));			
			push @{$tempAttributeMethodMatrix[$i]}, $sAttributeMethodMatrix->[$i][$j];				 		
		}
	}
	
	return 1 if (isDisjoint(\@tempAttributeMethodMatrix));
		
	return 0;
} # END sub isAGlueMethodSet


sub isDisjoint{
	my $sAttributeMethodMatrix = shift;
	
	my $noSubGraphs = 0;
	my @methodArray = ();
	my @attributeArray = ();
	
	$noSubGraphs = getNumberOfSubGraphs($sAttributeMethodMatrix, \@methodArray, \@attributeArray);
	
	return 1 if ($noSubGraphs > 1);
	return 0;
} # END sub isDisjoint
	

sub findElem{
	my $sArray = shift;
	my $sValue = shift;
	
	my $found = 0;	
	my $i = 0;
	
	while (!$found && $i < @{$sArray})
	{
		$found = 1 if ($sArray->[$i] == $sValue);
		$i++;
	}
	
	return $found;
} # End sub findElem


sub getNumberOfSubGraphs{
	#�ж�һ��ͼ�Ƿ�����ͨͼ
  #����: ���Է�������
  my $sAttributeMethodMatrix = shift;	 
  my $methodVisited = shift; 
  my $attributeVisited = shift; 
  
  
  #����ֵ: ����
  #��һ��������ʾ��ͼ�м�����ͨ��ͼ
  #�ڶ���������ʾÿ�������ڵ������ĸ���ͼ
  #������������ʾÿ�����Խڵ������ĸ���ͼ
  
  my $noOfSubGraphs = 0;
  
  my $row = @{$sAttributeMethodMatrix};
  my $col = @{$sAttributeMethodMatrix->[0]};
  
  for (my $attributeNode = 0; $attributeNode < $row; $attributeNode++){
  	$attributeVisited->[$attributeNode] = 0; #0��ʾû�б����ʹ�, ��Ȼ����ʾ�ýڵ����ڵڼ�����ͼ  	
  }

  for (my $methodNode = 0; $methodNode < $col; $methodNode++){
  	$methodVisited->[$methodNode] = 0; #0��ʾû�б����ʹ�, ��Ȼ����ʾ�ýڵ����ڵڼ�����ͼ  	
  }
 
  for (my $attributeNode = 0; $attributeNode < $row; $attributeNode++){
  	if ($attributeVisited->[$attributeNode] == 0){
  		$noOfSubGraphs++;
  		dfsForCBMC($sAttributeMethodMatrix, $attributeNode, "attribute", 
  		           $noOfSubGraphs, $methodVisited, $attributeVisited);
  	}
  }
 
  for (my $methodNode = 0; $methodNode < $col; $methodNode++){
  	if ($methodVisited->[$methodNode] == 0){
  		$noOfSubGraphs++;
  		dfsForCBMC($sAttributeMethodMatrix, $methodNode, "method", 
  		           $noOfSubGraphs, $methodVisited, $attributeVisited);
  	}
  }
 
  return $noOfSubGraphs;
} 


sub dfsForCBMC{
	#�����Է����������������ȱ���
	my $sAttributeMethodMatrix = shift;
	my $sCurrentNode = shift;  #��ǰ���ʵĽڵ�
	my $sNodeType = shift;     #��ǰ�ڵ������: ������������?
	my $sMark = shift;         #����ǰ�ڵ����ķ��ʱ��, ����ʹ����ͼ�ı��(��ͬ��ŵĽڵ�����ͬһ����ͼ)
	my $sMethodVisited = shift; #�洢�����Ƿ��Ѿ������ʵ���Ϣ
	my $sAttributeVisited = shift; #�洢�����Ƿ��Ѿ������ʵ���Ϣ
	

	if ($sNodeType eq "method"){
		$sMethodVisited->[$sCurrentNode] = $sMark;
#		print "method node => ", $sCurrentNode, "\n";
		for (my $row = 0; $row < @{$sAttributeMethodMatrix}; $row++){
			next if ($sAttributeMethodMatrix->[$row][$sCurrentNode] == 0); #�����ڽڵ�
			next if ($sAttributeVisited->[$row]);
			dfsForCBMC($sAttributeMethodMatrix, $row, "attribute", $sMark, $sMethodVisited, $sAttributeVisited);			
		} 		
	}
	
	if ($sNodeType eq "attribute"){
		$sAttributeVisited->[$sCurrentNode] = $sMark;
#		print "attribute node => ", $sCurrentNode, "\n";
		for (my $col = 0; $col < @{$sAttributeMethodMatrix->[0]}; $col++){
			next if ($sAttributeMethodMatrix->[$sCurrentNode][$col] == 0); #�����ڽڵ�
			next if ($sMethodVisited->[$col]);
			dfsForCBMC($sAttributeMethodMatrix, $col, "method", $sMark, $sMethodVisited, $sAttributeVisited);			
		} 		
	}
} # End sub dfsForCBMC

 
sub getSelectedSet{
	# �г��������������: ��$maxValue+1����������ѡ$noElementsInSet����
	my $sArrayOfSets = shift; #��Ž��
	my $sList = shift; #��ʱʹ�õĴ洢�ռ�
	my $sCurrentPosition = shift;
	my $sCurrentValue = shift;
	my $sNoElementsInSet = shift;
	my $sMaxNoElem = shift; #����Ԫ�ر��
	
	
	if ($sCurrentPosition >= $sNoElementsInSet){
		my $count = @{$sArrayOfSets};
		
		for (my $i = 0; $i < @{$sList}; $i++){
#			print $sList->[$i], "\t";
			$sArrayOfSets->[$count]->[$i] = $sList->[$i];
		}
		
#		print "\n";
		return 1;
	}
	
	
	for (my $value = $sCurrentValue; $value <= $sMaxNoElem; $value++){
		$sList->[$sCurrentPosition] = $value;
		getSelectedSet($sArrayOfSets, $sList, $sCurrentPosition + 1, $value + 1, $sNoElementsInSet, $sMaxNoElem);
	}
#	return 0;		
} # END sub getSlectedSet
 
 

#####����ICBMC�ĺ���#########

sub ICBMC{
	my $sAttributeMethodMatrix = shift;		
	
  print "computing ICBMC.....\n";
  		
	return 1 if (@{$sAttributeMethodMatrix} == 0); #ֻ��һ�����Խڵ���߷����ڵ��ͼ
	
	return 1 if (isMCC($sAttributeMethodMatrix));
	return 0 if (isDisjoint($sAttributeMethodMatrix));
	
	my @arrayOfGlueEdgeSet = getArrayOfGlueEdgeSet($sAttributeMethodMatrix);

  my $maxICBMC = 0;

	my $Fs = 0;
	my $Fc = 0;
	
  foreach my $currentGlueEdgeSet (@arrayOfGlueEdgeSet){    
  	
  	my $noSubGraphs = 0;
  	my @methodArray = ();   #��¼ÿ���������ڵڼ�����ͼ
  	my @attributeArray = ();  #��¼ÿ���������ڵڼ�����ͼ
  	
  	my @graphWithoutGlueEdges = excludeGlueEdges($sAttributeMethodMatrix, $currentGlueEdgeSet);  	
  	
  	$noSubGraphs = getNumberOfSubGraphs(\@graphWithoutGlueEdges, \@methodArray, \@attributeArray);
  	
  	my $sum;
  	
  	$sum = 0;
  	
  	for (my $i = 1; $i <= $noSubGraphs; $i++){
  		my @aSubGraph = getSubGraph(\@graphWithoutGlueEdges, $i, \@methodArray, \@attributeArray);  		
  		my $temp = ICBMC(\@aSubGraph); 		
  		$sum = $sum + $temp;
  	}
  	
  	$Fs = $sum / $noSubGraphs;  	
  	$Fc = @{$currentGlueEdgeSet} / (@methodArray * @attributeArray);
  	# getMaxNoInterEdgesBetweenSubGraphs(\@methodArray, \@attributeArray);  	
 	  	
  	$maxICBMC = $Fs * $Fc if ($Fs * $Fc > $maxICBMC);
  }  
  
  return $maxICBMC;	
} # END sub ICBMC


sub excludeGlueEdges{
	my $sAttributeMethodMatrix = shift;
	my $sCurrentGlueEdgeSet = shift;
	
	my @resultGraph = ();	
	
	for (my $row = 0; $row < @{$sAttributeMethodMatrix}; $row++){
		for (my $col = 0; $col < @{$sAttributeMethodMatrix->[0]}; $col++){
			$resultGraph[$row][$col] = $sAttributeMethodMatrix->[$row][$col];			
		}		
	}	
	
	for (my $i = 0; $i < @{$sCurrentGlueEdgeSet}; $i++){
		$resultGraph[$sCurrentGlueEdgeSet->[$i]->{"Row"}][$sCurrentGlueEdgeSet->[$i]->{"Col"}] = 0;		
	}
		
	return @resultGraph;
} # END sub excludeGlueEdges


sub getArrayOfGlueEdgeSet{
	#����:һ����ͨͼ
	#���: "��ˮ"�߼�����, ���ݽṹ: ��ά����, ��һά��ÿ��Ԫ�ض�Ӧ��һ����ˮ�߼� (һ����ͨͼ�����ж����ˮ������)
	#      �ڸ�����һάԪ�ص������, ÿ���ڶ�ά��Ԫ����һ��"��ˮ"��, ��ʵ����hash�������, ÿ��hash����2��Ԫ�����
	#      , keyΪ"Row"��"Col".
	
	
	my $sAttributeMethodMatrix = shift;	 
	my @arrayOfGlueEdgeSet = ();
		
	my $found = 0;	#�������һ����ˮ�߼�, �����˳�. ��Ϊ�����彺ˮ����ʹ����ͨͼ��Ϊ����ͨͼ����С������
	my $currentNoElemInGlueEdgeSet = 1;  #��ǰ�Ľ�ˮ�߼��а�����Ԫ����Ŀ
	
	my $countEdge = 0;
	my @allEdges = ();
	
	for (my $i = 0; $i < @{$sAttributeMethodMatrix}; $i++){
		for (my $j = 0; $j < @{$sAttributeMethodMatrix->[0]}; $j++){
			if ($sAttributeMethodMatrix->[$i][$j] == 1){
				$allEdges[$countEdge]->{"Row"} = $i;
				$allEdges[$countEdge]->{"Col"} = $j;				
				$countEdge++ ;
			}			
		}
	}
	
  my $maxNoElemInGlueEdgeSet = $countEdge; #��ˮ������Ԫ����Ŀ���ֵ	
    
	while (!$found && $currentNoElemInGlueEdgeSet <= $maxNoElemInGlueEdgeSet){
		 my $maxNoElem = $maxNoElemInGlueEdgeSet - 1; #����Ԫ�ر��, ��Ϊ����Ǵ�0��ʼ
		 
		 my @allArrayOfEdgeSet = ();
		 my @tempArray = ();
		 
     getSelectedSet(\@allArrayOfEdgeSet, \@tempArray, 0, 0, $currentNoElemInGlueEdgeSet, $maxNoElem);		
     
     foreach my $currentSet (@allArrayOfEdgeSet){     
     	  my @currentEdgeSet = ();
     	  for (my $i = 0; $i < @{$currentSet}; $i++){
     	  	$currentEdgeSet[$i]->{"Row"} = $allEdges[$currentSet->[$i]]->{"Row"};
     	  	$currentEdgeSet[$i]->{"Col"} = $allEdges[$currentSet->[$i]]->{"Col"};    
     	  }     	  
     	  
     	  if (isAGlueEdgeSet($sAttributeMethodMatrix, \@currentEdgeSet)){
     		   my $row = @arrayOfGlueEdgeSet;
     		   for (my $col = 0; $col < @currentEdgeSet; $col++){
     		   	  $arrayOfGlueEdgeSet[$row][$col]->{"Row"} = $currentEdgeSet[$col]->{"Row"}; 
     		   	  $arrayOfGlueEdgeSet[$row][$col]->{"Col"} = $currentEdgeSet[$col]->{"Col"}; 
     		   }#end for     		
     	  } #end if         	
     }#end for
     
     $found = 1 if (@arrayOfGlueEdgeSet);
     $currentNoElemInGlueEdgeSet++;	
	}
			
	return @arrayOfGlueEdgeSet;	
} # END sub getArrayOfGlueEdgeSet
 

sub isAGlueEdgeSet{
	my $sAttributeMethodMatrix = shift;	 
	my $sCurrentSet = shift;	
		
	my @tempAttributeMethodMatrix = ();
	
	my $row = @{$sAttributeMethodMatrix};
	my $col = @{$sAttributeMethodMatrix->[0]};

	for (my $i = 0; $i < $row; $i++){		
		for (my $j = 0; $j < $col; $j++){
			$tempAttributeMethodMatrix[$i][$j] = $sAttributeMethodMatrix->[$i][$j];				 		
		}
	}
	
	for (my $i = 0; $i < @{$sCurrentSet}; $i++){
		$tempAttributeMethodMatrix[$sCurrentSet->[$i]->{"Row"}][$sCurrentSet->[$i]->{"Col"}] = 0;
	}		
	
	my $noSubGraphs = 0;
	my @methodArray = ();
	my @attributeArray = ();
	
	$noSubGraphs = getNumberOfSubGraphs(\@tempAttributeMethodMatrix, \@methodArray, \@attributeArray);	
		
	return 0 if ($noSubGraphs == 1);
	
	#��ÿ����ͼ, ��¼��ڵ���(��������������)
	my @noNodesubGraph = ();
	
	for (my $i = 0; $i < $noSubGraphs; $i++){
		my $elem = $i + 1;
		$noNodesubGraph[$i] = countOccurence(\@methodArray, $elem) + countOccurence(\@attributeArray, $elem);	
	}
	
	
	#��ͼ�ڵ�������Сֵ
	my $minNode = $noNodesubGraph[0];
	for (my $i = 1; $i < $noSubGraphs; $i++){
		$minNode = $noNodesubGraph[$i] if ($noNodesubGraph[$i] < $minNode);
	}
	
	return 1 if ($minNode > 1);  #����ǽ�ˮ�߼�, ��ͼ���ٰ��������ڵ� 	
		
	return 0;
} # END sub isAGlueEdgeSet



sub getMaxNoInterEdgesBetweenSubGraphs{
	my $sMethodArray = shift; #��¼ÿ���������ڵڼ�����ͼ, ע����С����ͼ�����1
	my $sAttributeArray = shift; #��¼ÿ���������ڵڼ�����ͼ
	
	my $noSubGraph = 0;
	
	for (my $i = 0; $i < @{$sMethodArray}; $i++){
		$noSubGraph = $sMethodArray->[$i] if ($sMethodArray->[$i] > $noSubGraph);
	}
	
	for (my $i = 0; $i < @{$sAttributeArray}; $i++){
		$noSubGraph = $sAttributeArray->[$i] if ($sAttributeArray->[$i] > $noSubGraph);
	}
	
	
	#��ÿ����ͼ, ��¼�䷽������������
	my @subGraphInfo = ();
	
	for (my $i = 0; $i < $noSubGraph; $i++){
		$subGraphInfo[$i]->{"NoOfMethods"} = countOccurence($sMethodArray, $i+1);
		$subGraphInfo[$i]->{"NoOfAttributes"} = countOccurence($sAttributeArray, $i+1);	
	}
	
	my $sum = 0;
	
	for (my $i = 0; $i < $noSubGraph; $i++){
		for (my $j = 0; $j < $noSubGraph; $j++){
			next if ($i == $j);
			$sum = $sum + $subGraphInfo[$i]->{"NoOfMethods"} * $subGraphInfo[$j]->{"NoOfAttributes"};			
		}
	}
		
	return $sum;	
} # END sub getMaxNoInterEdgesBetweenSubGraphs


sub countOccurence{
	my $array = shift;
	my $elem = shift;
	
	my $count = 0;	
	
	for (my $i = 0; $i < @{$array}; $i++){
		$count++ if ($array->[$i] == $elem);	
	}
	
	return $count;
} # END sub countOccurence



#########����CAMCϵ��#################### 
 
sub CAMC{
	 #----����CAMC��CAMCs-----
   my $sParameterTypeMethodMatrix = shift;   

  if (@{$sParameterTypeMethodMatrix} == 0){ #ʵ����������������ܳ���, ��Ϊǰ�������ʹ��������һ������
   	return (0, 0);
  }   
   
   my $sum = 0;   
   my $noRow = @{$sParameterTypeMethodMatrix};
   my $noCol = @{$sParameterTypeMethodMatrix->[0]};
   
   return wantarray?("Undefined","Undefined"):"Undefined" if ($noCol == 0);
   return wantarray?(1, 1):1 if ($noCol == 1);
   
   for (my $i = 0; $i < $noRow; $i++){
   	for (my $j = 0; $j < $noCol; $j++){
   		$sum++ if ($sParameterTypeMethodMatrix->[$i][$j] == 1);
   	}
   }
   
   my $sCAMC = $sum / ($noRow * $noCol);
   my $sCAMCs = ($sum + $noCol) / (($noRow + 1)*$noCol);
   
   return wantarray?($sCAMC, $sCAMCs): $sCAMC;
} # END sub CAMC


sub NHD{
   #----����NHD��NHDs-----
   my $sParameterTypeMethodMatrix = shift;   

  if (@{$sParameterTypeMethodMatrix} == 0){ #ʵ����������������ܳ���, ��Ϊǰ�������ʹ��������һ������
   	return (0, 0);
  }   
   
   my $sum = 0;
   my $noRow = @{$sParameterTypeMethodMatrix};
   my $noCol = @{$sParameterTypeMethodMatrix->[0]};

   return wantarray?("Undefined","Undefined"):"Undefined" if ($noCol == 0);
   return wantarray?(1, 1):1 if ($noCol == 1);
   
   my @cc;
   
   for (my $i = 0; $i < $noRow; $i++){
   	$cc[$i] = 0;
   	for (my $j = 0; $j < $noCol; $j++){
   		$cc[$i]++ if ($sParameterTypeMethodMatrix->[$i][$j] == 1);
   	}
   }
   
   for (my $i = 0; $i < $noRow; $i++){
   	$sum = $sum + $cc[$i]*($noCol - $cc[$i]);  	
   }
   
   my $sNHD;
   my $sNHDs;
   
   $sNHD = 1 - 2 * $sum / ($noRow * $noCol * ($noCol - 1));
   $sNHDs = 1 - 2 * $sum / (($noRow + 1) * $noCol * ($noCol - 1));
   
   return wantarray?($sNHD, $sNHDs): $sNHD;
 } # END sub NHD
   
   
sub SNHD{
   #----����SNHD-----
   my $sParameterTypeMethodMatrix = shift;     

  if (@{$sParameterTypeMethodMatrix} == 0){#ʵ����������������ܳ���, ��Ϊǰ�������ʹ��������һ������
   	return 0;
  }   
   
   my $NHDmin;
   my $NHDmax;
   my $NHD = NHD($sParameterTypeMethodMatrix);
   my $sSNHD;
   
   
   my $noRow = @{$sParameterTypeMethodMatrix};
   my $noCol = @{$sParameterTypeMethodMatrix->[0]};
      
   my $sum = 0;
   for (my $i = 0; $i < $noRow; $i++){
   	for (my $j = 0; $j < $noCol; $j++){
   		$sum = $sum + 1 if ($sParameterTypeMethodMatrix->[$i][$j] == 1);  		
   	}  	
   }
       
   if ($noCol == 0){    
   	$NHDmin = "Undefined";
   	$NHDmax = "Undefined";  		
    $sSNHD = "Undefined";  	   	
   }
   elsif ($noCol == 1){
   	$NHDmin = 1;
   	$NHDmax = 1;  		
    $sSNHD = 1;  	
   }
   else{
   	my $dd = int($sum / $noRow);
    my $qq = $sum % $noRow;
    my $cc = int(($sum - $noRow) / ($noCol - 1));
    my $rr = ($sum - $noRow) % ($noCol - 1);
    
   	$NHDmin = 1 - 2*($qq*($dd+1)*($noCol-$dd-1) + ($noRow-$qq)*$dd*($noCol-$dd)) / ($noRow*$noCol*($noCol-1));
   	$NHDmax = 1 - 2*(($rr+1)*($noCol-$rr-1) + ($noRow-$cc-1)*($noCol-1)) / ($noRow*$noCol*($noCol-1));  	
    if (($NHDmin == $NHDmax) && ($sum < $noRow * $noCol)){
   	  $sSNHD = 0;
    }
    elsif ($sum == $noRow * $noCol){
   	  $sSNHD = 1;
    }
    else {
   	  $sSNHD = 2 * ($NHD - $NHDmin) / ($NHDmax - $NHDmin) - 1; 	
    }  	
   }  
   
#   print "NHDmin = ", $NHDmin, "\n";
#   print "NHDmax = ", $NHDmax, "\n";

   return $sSNHD; 
} # END sub SNHD


sub SNHDs{
   #----����SNHDs-----
   my $sParameterTypeMethodMatrix = shift;  	
   
   my $noRow = @{$sParameterTypeMethodMatrix};
   my $noCol = @{$sParameterTypeMethodMatrix->[0]};
   
   my @tempArr;
   
   for (my $i = 0; $i < $noRow; $i++){
   	 for (my $j = 0; $j < $noCol; $j++){
   	 	$tempArr[$i][$j] = $sParameterTypeMethodMatrix->[$i][$j];   	 	
   	 }   	
   }
      
   for (my $j = 0; $j < $noCol; $j++){ #��ÿ��������һ����ͬ�Ĳ���"self"
   	 $tempArr[$noRow][$j] = 1;   	
   }
   
   my $sSNHDs = SNHD(\@tempArr);  
	
	 return $sSNHDs;	
}



###############compute MI########################

# return declaration ref (based on language) or 0 if unknown
sub getDeclRef 
{
    my ($ent) =@_;
    my $decl;
    return $decl unless defined ($ent);
    
   ($decl) = $ent->refs("definein","",1);
	 ($decl) = $ent->refs("declarein","",1) unless ($decl);

    return $decl;
} # END sub getDeclRef


# scan the code in the specified range of lines
# and return the 4 basic operator/operand metrics
sub scanEntity
{
  my ($lexer, $startline, $endline) = @_;
  my $n1=0;
  my $n2=0;
  my $N1=0;
  my $N2=0;
  
  my %n1 = ();
  my %n2 = ();


  foreach my $lexeme ($lexer->lexemes($startline,$endline)) 
  {

     if (($lexeme->token eq "Operator") || ($lexeme->token eq "Keyword") || ($lexeme->token eq "Punctuation"))
     {  
        
        if ($lexeme->text() !~ /[)}\]]/)
        {
           $n1{$lexeme->text()} = 1;

#           print "\t  n1--->", $lexeme->text(), "\n";

           $N1++;
        }
     }
     elsif (($lexeme->token eq "Identifier") || ($lexeme->token eq "Literal") || ($lexeme->token eq "String"))
     {
        $n2{$lexeme->text()} = 1;

#        print "\t  n2--->", $lexeme->text(), "\n";

        $N2++;
     }
  }
  
  $n1 = scalar( keys(%n1));
  $n2 = scalar( keys(%n2));  
   
  return ($n1,$n2,$N1,$N2);
} # End sub scanEntity





# return array of functions in a file
sub getFuncs {
    my $db = shift;
    my $file = shift;
    my $lexer = shift;
    my $language = $db->language();   # use language of $file when available
    my @funcs = ();

    my $refkind;
    my $entkind;
    if ($language =~ /ada/i) {
	$refkind = "declarein body";
	$entkind = "function,procedure";
    } elsif ($language =~ /java/i) {
	$refkind = "definein";
	$entkind = "method";
    } elsif ($language =~ /c/i) {
	$refkind = "definein";
	$entkind = "function";
    } else {
	return ();
    }

    $lexer = $file->lexer() if !$lexer;
    foreach my $lexeme ($lexer->lexemes()) {
	next if !$lexeme;
	my $ref = $lexeme->ref();
	my $ent = $lexeme->entity();
	if ($ref && $ent && $ref->kind->check($refkind) && $ent->kind->check($entkind)) {
	    push @funcs, $ent;
	}
    }
    return @funcs;
} # END sub getFuncs



sub isCPlusPlusConstructor{
	my $class = shift;
	my $method = shift;	
	
	if ($class->name() eq $method->name()){
#		print "Consructor: ", $method->name(), "  ",$method->id(), " \n";
		return 1;
	}	
	return 0;
} # END sub isCPlusPlusConstructor


sub isCPlusPlusDestructor{
	my $class = shift;
	my $method = shift;
	if ("~".$class->name() eq $method->name()){
#		print "Desructor: ", $method->name(), "  ",$method->id(), " \n";
		return 1;	
	}
	return 0;
} # END sub isCPlusPlusDestructor


sub isCPlusPlusAccessOrDelegationMethod{
	my $class = shift;
	my $method = shift;
	
#	print "Yes, I am here!\n";
	
	my %readList = ();
	foreach my $attribute ($method->ents("Use","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$readList{$attribute->id()} = 1;
	}
	
	my %writeList = ();
	foreach my $attribute ($method->ents("Set","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$writeList{$attribute->id()} = 1;
	}
	
	my %modifyList = ();
	foreach my $attribute ($method->ents("Modify","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$modifyList{$attribute->id()} = 1
	}
	
	my %allAttributeList = ();
	foreach my $key (keys %readList){
		$allAttributeList{$key} = 1;
	}

	foreach my $key (keys %writeList){
		$allAttributeList{$key} = 1;
	}

	foreach my $key (keys %modifyList){
		$allAttributeList{$key} = 1;
	}
	
	my $noModifyAttribute = scalar (keys %modifyList);
	my $noAllAccessAttribute = scalar (keys %allAttributeList);
	my $noStatements = $method->metric("CountStmt");	

	my @callList = $method->ents("Call ~inactive, use ptr ~inactive","function,  method");	
	my $noCalls = @callList;  

#	print $method->name(), "  ",$method->id(), " \n";
#	print "\t\t noStatements = ", $noStatements, "\n";
#	print "\t\t noAllAccessAttribute = ", $noAllAccessAttribute, " \n";	

	if ($noStatements == 1 && $noAllAccessAttribute ==1){
		if ($noCalls){
#   		print "Delegation method: ", $method->name(), "  ",$method->id(), " \n";
    	return 1;  						
		}
# 		print "Access method: ", $method->name(), "  ",$method->id(), " \n";
   	return 1;  					
	}
  
  return 0;   	
} # END sub isCPlusPlusAccessOrDelegationMethod


sub isJavaConstructor{
	my $class = shift;
	my $method = shift;
	my @className = split /\./, $class->name();
	my @methodName = split /\./, $method->name();
	
	if ($className[$#className] eq $methodName[$#methodName]){	
#		print $class->name(), "<============>",$method->name(), " \n";
		
		return 1;
	}
	return 0;	
} # END sub isJavaConstructor


sub isJavaDestructor{

} # END sub isJavaDestructor


sub isJavaAccessOrDelegationMethod{
	my $class = shift;
	my $method = shift;  
  
	my %readList = ();
	foreach my $attribute ($method->ents("Use","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$readList{$attribute->id()} = 1;
	}
	
	my %writeList = ();
	foreach my $attribute ($method->ents("Set","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$writeList{$attribute->id()} = 1;
	}
	
	my %modifyList = ();
	foreach my $attribute ($method->ents("Modify","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved")){
		$modifyList{$attribute->id()} = 1
	}
	
	my %allAttributeList = ();
	foreach my $key (keys %readList){
		$allAttributeList{$key} = 1;
	}

	foreach my $key (keys %writeList){
		$allAttributeList{$key} = 1;
	}

	foreach my $key (keys %modifyList){
		$allAttributeList{$key} = 1;
	}
	
	my $noModifyAttribute = scalar (keys %modifyList);
	my $noAllAccessAttribute = scalar (keys %allAttributeList);
	my $noStatements = $method->metric("CountStmt");	

#	$CountLineCode = $class->metric("CountLineCode");
#	
#	$CountLineExe = $class->metric("CountLineExe");


	my @callList = $method->ents("Call ~inactive, use ptr ~inactive","function,  method");	
	my $noCalls = @callList;  
	
#	print $method->name(), "  ",$method->id(), " \n";
#	print "\t\t noStatements = ", $noStatements, "\n";
#	print "\t\t noAllAccessAttribute = ", $noAllAccessAttribute, " \n";

	if ($noStatements == 2 && $noAllAccessAttribute ==1){  #Java�������㷽ʽ��C++��ϸ΢�Ĳ��, ������2
		if ($noCalls){
#   		print "Delegation method: ", $method->name(), "  ",$method->id(), " \n";
    	return 1;  						
		}
# 		print "Access method: ", $method->name(), "  ",$method->id(), " \n";
   	return 1;  					
	}
  
  return 0;   	
} # END sub isJavaAccessOrDelegationMethod


sub hashTable2Matrix{
	 #ע��: ��һ���������������ı����Ϣ
	 #�����ÿ��������һ����ϣ�������
	 #      ����ֵ��һ���б�ʾ����(����),�б�ʾ����, ֵ������ʹ�ϵ�ľ���
	 
#   print "-----From hashTable2Matrix--------\n";	 
	 
	 	 	 
	 my $sNoForMethod = shift(@_);

   #����Ҫ,��֤���	 
	 foreach my $key (sort keys %{$sNoForMethod}){
	  	delete $sNoForMethod->{$key};
	 }	 
	
	 
   my %hashTable = ();
   
   #�������ϣ��ϲ���һ����ϣ��hashTable
   foreach my $aHashRef (@_){   	 
   	foreach my $attributeKey (sort keys %{$aHashRef}){
   		my %tempMethodHashTable = %{$aHashRef->{$attributeKey}};
   		foreach my $methodKey (sort keys %tempMethodHashTable){
   			$hashTable{$attributeKey}->{$methodKey} = 1;
   		}
   	}	
   }
   
   my $NumOfRow = 0;
   my $NumOfColumn = 0;


   #������������ 
   my @tempArr;
   @tempArr = (keys %hashTable);
   $NumOfRow = @tempArr;

   #�ж��Ƿ���keyΪ"withoutParameterAndAttribute"�������: ��Щ�����������κ�����(����)   
   if (exists $hashTable{"withoutParameterAndAttribute"}){
    	$NumOfRow = $NumOfRow - 1;
   }
     
#   print "NumOfRow: ", $NumOfRow, "\n";
      
   #������������  
   
   foreach my $attributeKey (sort keys %hashTable){
   	my %tempMethodHashTable = %{$hashTable{$attributeKey}};
   	foreach my $methodKey (sort keys %tempMethodHashTable){
   		next if ($methodKey eq "withoutMethodAccess");
   		$sNoForMethod->{$methodKey} = 1;   		
   	}
   }
   
   
  
   
   #��ÿ���������
   my $jj=0;
   foreach my $method (sort keys %{$sNoForMethod}){
   	$sNoForMethod->{$method} = $jj;
   	$jj++;  	
   } 
   
      
   @tempArr = (keys %{$sNoForMethod});
   $NumOfColumn = @tempArr;
#   print "NumOfColumn: ", $NumOfColumn, "\n";
   
   
   #��������
   my @outputMatrix;
   
   for (my $i = 0; $i < $NumOfRow; $i++)   {
   	for (my $j = 0; $j < $NumOfColumn; $j++){
   		$outputMatrix[$i][$j] = 0;   	
    }
   }
   
   my $i = 0;
   foreach my $attributeKey (sort keys %hashTable)  {
   	    next if ($attributeKey eq "withoutParameterAndAttribute");
   	    my %tempMethodHashTable = %{$hashTable{$attributeKey}};
   	    foreach my $methodKey (sort keys %tempMethodHashTable){
   	    	#����"withoutMethodAccess", ��������һ��������, ֻ��������ʾ��Ӧ������(����)û���κη�������
   	    	next if ($methodKey eq "withoutMethodAccess");
   	    	$outputMatrix[$i][$sNoForMethod->{$methodKey}] = 1;    	  
   	    }
    	  $i++;  	  
   }
   
 
   return wantarray?(\@outputMatrix, $NumOfRow, $NumOfColumn): \@outputMatrix;
} # END sub hashTable2Matrix


	
sub generateMethodMethodMatrix{
	#����:����-��������
	#����ֵ: ����-��������, ʵ�����Ƿ������ƾ���
	
	my $inputMatrix = shift;	
	my $sNoForMethods = shift; #�����ı��hash��
	my $includeMethodCall = shift; #����������Ƿ���������ù�ϵ
	                               #ֵΪ1ʱ,��ʾֱ�ӵı����ù�ϵ
	                               #ֵΪ2ʱ,��ʾ��ӵı����ù�ϵ  
	my $sMethodCallBySet = shift; #�����ù�ϵ����
	
	my @outputMatrix;
	
	#�������û������, ��������ֵ��Ϊ0�ķ�������
	if (@{$inputMatrix} == 0){		
		my $noRow = scalar (keys %{$sNoForMethods});
		my $noCol = $noRow;
		
		for (my $i = 0; $i < $noRow; $i++){
			for (my $j = 0; $j < $noCol; $j++){
				$outputMatrix[$i][$j] = 0;
			}
		}
		
		if ($includeMethodCall == 1){
			foreach my $fstKey (sort keys %{$sMethodCallBySet}){
				next if (!exists $sNoForMethods->{$fstKey});
				foreach my $sndKey (sort keys %{$sMethodCallBySet->{$fstKey}}){
					next if (!exists $sNoForMethods->{$sndKey});
					$outputMatrix[$sNoForMethods->{$fstKey}][$sNoForMethods->{$sndKey}] = 1;
					$outputMatrix[$sNoForMethods->{$sndKey}][$sNoForMethods->{$fstKey}] = 1;					
				}
			}			
		}	
		
		if ($includeMethodCall == 2){
			foreach my $fstKey (sort keys %{$sMethodCallBySet}){
				my @methodArray = (sort keys %{$sMethodCallBySet->{$fstKey}});
				for (my $i = 0; $i < @methodArray - 1; $i++){
					next if (!exists $sNoForMethods->{$methodArray[$i]});
					for (my $j = $i + 1; $j < @methodArray; $j++){
						next if (!exists $sNoForMethods->{$methodArray[$j]});
						$outputMatrix[$sNoForMethods->{$methodArray[$i]}][$sNoForMethods->{$methodArray[$j]}] = 1;
						$outputMatrix[$sNoForMethods->{$methodArray[$j]}][$sNoForMethods->{$methodArray[$i]}] = 1;						
					}
				}
			}			
		}
				
		return @outputMatrix;		
	}
			
	my $noRowOrColumn = @{$inputMatrix->[0]};
	
	for (my $i = 0; $i < $noRowOrColumn; $i++){
		for (my $j = 0; $j < $noRowOrColumn; $j++){
			$outputMatrix[$i][$j] = 0;
		}
	}
	
	for (my $i = 0; $i < $noRowOrColumn - 1; $i++){
		for (my $j = $i + 1; $j < $noRowOrColumn; $j++){
			
			my @arrayOne; 		
			my @arrayTwo;
			for (my $k = 0; $k < @{$inputMatrix}; $k++){
				$arrayOne[$k] = $inputMatrix->[$k][$i];
				$arrayTwo[$k] = $inputMatrix->[$k][$j];
			}
			
			if (isMethodSimilar(\@arrayOne, \@arrayTwo)){
				$outputMatrix[$i][$j] = 1;
				$outputMatrix[$j][$i] = 1;
			}			
		}		
	}
	
	if ($includeMethodCall == 1){
		foreach my $fstKey (sort keys %{$sMethodCallBySet}){
			next if (!exists $sNoForMethods->{$fstKey});				
			foreach my $sndKey (sort keys %{$sMethodCallBySet->{$fstKey}}){
				next if (!exists $sNoForMethods->{$sndKey});
				$outputMatrix[$sNoForMethods->{$fstKey}][$sNoForMethods->{$sndKey}] = 1;
				$outputMatrix[$sNoForMethods->{$sndKey}][$sNoForMethods->{$fstKey}] = 1;					
			}
		}			
	}	
	
	if ($includeMethodCall == 2){
			foreach my $fstKey (sort keys %{$sMethodCallBySet}){
				my @methodArray = (sort keys %{$sMethodCallBySet->{$fstKey}});
				for (my $i = 0; $i < @methodArray - 1; $i++){
					next if (!exists $sNoForMethods->{$methodArray[$i]});
					for (my $j = $i + 1; $j < @methodArray; $j++){
						next if (!exists $sNoForMethods->{$methodArray[$j]});
						$outputMatrix[$sNoForMethods->{$methodArray[$i]}][$sNoForMethods->{$methodArray[$j]}] = 1;
						$outputMatrix[$sNoForMethods->{$methodArray[$j]}][$sNoForMethods->{$methodArray[$i]}] = 1;						
				}
			}
		}			
	}
		
  return @outputMatrix;
} # END sub generateMethodMethodMatrix



sub depthFirstSearch{
	#����1:���������Ծ���
	#����2:��ʼ����
	#����3:��Ǿ���
	#����:ֱ���޸ı�Ǿ���
  
  my $aTwoDimArray = shift;
  my $node = shift; 
  my $visited = shift;
  my $noElem = shift;
  
  $visited->[$node] = 1;
    
  for (my $i = 0; $i < @{$aTwoDimArray}; $i++){
  	if (($aTwoDimArray->[$node][$i] == 1) && !$visited->[$i]){
  		depthFirstSearch($aTwoDimArray, $i, $visited);		
  	}
  }  
} # END sub depthFirstSearch




sub buildParameterHashTable{
	my $class = shift; #��
	my $excludePrivateProtectedMethods = shift; #�Ƿ��ų�"˽��"����"�ܱ���"����
	my $excludeConstructorAndDestructor = shift; #�Ƿ��ų�"���캯������������"
	my $excludeAccessAndDelegationMethod = shift; #�Ƿ��ų�"���ʷ����ʹ�����"
	my $includeFunctionType = shift; #�ڼ��㺯���Ĳ�������ʱ�Ƿ���������ķ���ֵ����
	my $sParaTable = shift;


  my @methodArray = ();
  
  if (!$excludePrivateProtectedMethods){
  	@methodArray = getRefsInClass($class, "define","function ~unknown,  method ~unknown");
  }
  else{
  	@methodArray = getRefsInClass($class, "define","function ~private ~protected ~unknown, method ~private ~protected ~unknown");
  }
  
  if (@methodArray == 0){
  	return 0; 	
  }
  

  #ĳЩhash�����
  foreach my $key (sort keys %{$sParaTable}){
  	delete $sParaTable->{$key};
  }
  

	#������
  foreach my $method (@methodArray){
	  my $func = $method->ent();
	  
 	  if ($excludeConstructorAndDestructor){
	  	next if $isConstructor->($class, $func);
	  	next if $isDestructor->($class, $func);
	  }	  

 	  if ($excludeAccessAndDelegationMethod){
# 	  	print "hallo!\n";
	  	next if $isAccessOrDelegationMethod->($class, $func);
	  }	  

	  
	  my @paraArray = $func->ents("Define","Parameter");
	  
	  if ($includeFunctionType && $func->type() && ($func->type() ne "void")){	  
	  	push @paraArray, $func; 
	  }
  
    if (@paraArray == 0){
	  	$sParaTable->{"withoutParameterAndAttribute"}->{$func->id()} = 1;
	  }
	  else{
	  	foreach my $param (@paraArray)  {  
    	  $sParaTable->{$param->type()}->{$func->id()} = 1;    	  
    	 } 
    }   
    
} #foreach $func
	
#�ų����ⷽ����, �п���û���κη���    
 if (scalar (keys %{$sParaTable}) == 0){
  	return 0;
 }	    
	    
return 1;		
} # END sub buildParameterHashTable
	
	

sub buildAttributeHashTables{
	my $class = shift; #��
	my $excludePrivateProtectedMethods = shift; #�Ƿ��ų�"˽��"����"�ܱ���"����
	my $excludeConstructorAndDestructor = shift; #�Ƿ��ų�"���캯������������"
	my $excludeAccessAndDelegationMethod = shift; #�Ƿ��ų�"���ʷ����ʹ�����"
	my $includeIndirectAccess = shift; #�Ƿ�������������Լ��"��ӷ���"��ϵ
	my $sAttributeReadTable = shift;
	my $sAttributeWriteTable = shift;
	my $sAttributeModifyTable = shift;
	my $sMethodWithoutAttributeParaTable = shift; #ֻ��һ��key(��withoutParameterAndAttribute)�Ĺ�ϣ��, ֵΪû�в����������Է��ʵķ���
	my $sAttributeWithoutAccessTable = shift; #��Ӧÿ��key, ֵΪ"withoutMethodAccess". ��ʾ������û���κη�������
	my $sDirectCallByMethodSet = shift; #ֱ�ӱ����õķ�����


  my @methodArray = ();
  @methodArray = getRefsInClass($class, "define","function ~unknown ~unresolved,  method ~unknown ~unresolved");
#  print "\n", "methodArray = ", scalar @methodArray, "\n";
  
  if (@methodArray == 0){
  	return 0; 	
  }
    
  my %classMethodTable = ();
  for (my $i = 0; $i < @methodArray; $i++){
   	$classMethodTable{$methodArray[$i]->ent()->id()} = 1;
  }
  
 
  my %classAttributeTable = ();  
  my @classAttributes = getRefsInClass($class, "define","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");  
  for (my $i = 0; $i < @classAttributes; $i++){
   	$classAttributeTable{$classAttributes[$i]->ent()->id()} = 1;
  }


  foreach my $key (sort keys %{$sAttributeReadTable}){
  	delete $sAttributeReadTable->{$key};
  }
  
  foreach my $key (sort keys %{$sAttributeWriteTable}){
  	delete $sAttributeWriteTable->{$key};
  }
	
  foreach my $key (sort keys %{$sAttributeModifyTable}){
  	delete $sAttributeModifyTable->{$key};
  }
	
  foreach my $key (sort keys %{$sMethodWithoutAttributeParaTable}){
  	delete $sMethodWithoutAttributeParaTable->{$key};
  }

  foreach my $key (sort keys %{$sAttributeWithoutAccessTable}){
  	delete $sAttributeWithoutAccessTable->{$key};
  }

  foreach my $key (sort keys %{$sDirectCallByMethodSet}){
  	delete $sDirectCallByMethodSet->{$key};
  }  


  foreach my $method (@methodArray){
#	  print "**********method = ", $func->name(), "***********\n";
	  my $func = $method->ent();
	  
 	  if ($excludeConstructorAndDestructor){ 	  	
	  	next if $isConstructor->($class, $func);
	  	next if $isDestructor->($class, $func);
	  }	  

 	  if ($excludeAccessAndDelegationMethod){
	  	next if $isAccessOrDelegationMethod->($class, $func);
	  }	  



	  my @callMethodSet = $func->ents("Call ~inactive, use ptr ~inactive","function ~unknown ~unresolved,  method ~unknown ~unresolved");	  
	  foreach my $tempMethod (@callMethodSet){
	  	next if (!exists $classMethodTable{$tempMethod->id()});	  	  	
	  	$sDirectCallByMethodSet->{$tempMethod->id()}->{$func->id()} = 1;	    
	  }
	}
	
	  
	my %tempInDirectCallByMethodSet = getIndirectCallByMethodSet($sDirectCallByMethodSet);   


	#������
  foreach my $method (@methodArray){
	  my $func = $method->ent();
#	  print "**********method = ", $func->name(), "***********\n";	  
 	  if ($excludeConstructorAndDestructor){
	  	next if $isConstructor->($class, $func);
	  	next if $isDestructor->($class, $func);
	  }	  

 	  if ($excludeAccessAndDelegationMethod){
	  	next if $isAccessOrDelegationMethod->($class, $func);
	  }	  
	  
	  
	  
	  
#	  my $refObjects = $func->ref()->file()->lexer();
#	  
#	  my ($startref) = $func->refs("definein", "", 1);
#	  my ($endref) = $func->refs("end", "", 1);
#	  
#	  my @tokenArray = $refObjects->lexemes($startref->line(), $endref->line());
#	  if (!@tokenArray){
#	  	print "has no tokens \n";
#	  }
#	  foreach my $refO (@tokenArray) {
#	  	print "token = ", $refO->token(), "\t text = ", $refO->text();
#	  	if (defined($refO->ent())){
#	  		print "\t ent type = ", $refO->ent()->type();
#	  	}
#	  	print "\n";	  	
#	  }
	  
	  
	  
#	  print "\n\n";
	  
  
  	foreach my $attribute ($func->ents("Use","Member Object ~unknown ~unresolved ~Local, Member Variable ~unknown ~unresolved ~Local"))  {
  		  next if (!exists $classAttributeTable{$attribute->id()}); 
        $sAttributeReadTable->{$attribute->id()}->{$func->id()} = 1;
        
        if ($includeIndirectAccess && exists $tempInDirectCallByMethodSet{$func->id()}){
        	foreach my $methodKey (sort keys %{$tempInDirectCallByMethodSet{$func->id()}}){
        		$sAttributeReadTable->{$attribute->id()}->{$methodKey} = 1;        		
        	}         	
        }        
      #  print "Reading Attribute====>", $attribute->longname(), ":::::::", $attribute->kindname(), "\n";
     }
    
    
    
    
    foreach my $attribute ($func->ents("Modify","Member Object ~unknown ~unresolved ~Local, Member Variable ~unknown ~unresolved ~Local"))  {
    	  next if (!exists $classAttributeTable{$attribute->id()});
        $sAttributeModifyTable->{$attribute->id()}->{$func->id()} = 1;
        
        if ($includeIndirectAccess && exists $tempInDirectCallByMethodSet{$func->id()}){
        	foreach my $methodKey (sort keys %{$tempInDirectCallByMethodSet{$func->id()}}){
        		$sAttributeModifyTable->{$attribute->id()}->{$methodKey} = 1;        		
        	}         	
        } 
#       print "Modifying Attribute====>", $attribute->name(),  ":::::::", $attribute->kindname(), "\n";
    }
    

    foreach my $attribute ($func->ents("Set","Member Object ~unknown ~unresolved ~Local, Member Variable ~unknown ~unresolved ~Local"))  {
    	  next if (!exists $classAttributeTable{$attribute->id()});
        $sAttributeWriteTable->{$attribute->id()}->{$func->id()} = 1;
        
        if ($includeIndirectAccess && exists $tempInDirectCallByMethodSet{$func->id()}){
        	foreach my $methodKey (sort keys %{$tempInDirectCallByMethodSet{$func->id()}}){
        		$sAttributeWriteTable->{$attribute->id()}->{$methodKey} = 1;        		
        	}         	
        }         
#        print "Writing Attribute====>", $attribute->name(),  ":::::::", $attribute->kindname(), "\n";
    }
    
#    print "\n\n";
} #foreach $func


   if ($excludePrivateProtectedMethods){  
   	my @tempMethodArray = getRefsInClass($class, "define","function ~private ~protected ~unknown ~unresolved, method ~private ~protected ~unknown ~unresolved");
   	
   	my @pubMethodArray = ();
   	for (my $i = 0; $i < @tempMethodArray; $i++){
   		$pubMethodArray[$i] = $tempMethodArray[$i]->ent()->id();
   	}
   	
   	includeOnlyElements($sAttributeReadTable, \@pubMethodArray);
   	includeOnlyElements($sAttributeWriteTable, \@pubMethodArray);
   	includeOnlyElements($sAttributeModifyTable, \@pubMethodArray);   
   }


    #�ռ�û�з����κ����Եķ���
    my %tempMethodHashTable;
    
    foreach my $method (@methodArray){    	  
	     foreach my $fstKey (keys %{$sAttributeReadTable}){
	      foreach my $sndKey (keys %{$sAttributeReadTable->{$fstKey}}){
	     		$tempMethodHashTable{$sndKey} = 1;
	     	}
       }

	     foreach my $fstKey (keys %{$sAttributeWriteTable}){
	  	   foreach my $sndKey (keys %{$sAttributeWriteTable->{$fstKey}}){
	  		   $tempMethodHashTable{$sndKey} = 1;
	  	   }
       }
    
	     foreach my $fstKey (keys %{$sAttributeModifyTable}){
	  	   foreach my $sndKey (keys %{$sAttributeModifyTable->{$fstKey}}){
	  		   $tempMethodHashTable{$sndKey} = 1;
	  	   }
       }
    }
    
    foreach my $method (@methodArray){
    	my $func = $method->ent();
    	
 	  if ($excludeConstructorAndDestructor){
	  	next if $isConstructor->($class, $func);
	  	next if $isDestructor->($class, $func);
	  }	  

 	  if ($excludeAccessAndDelegationMethod){
	  	next if $isAccessOrDelegationMethod->($class, $func);
	  }	  
  
    	    
    	if (!exists $tempMethodHashTable{$method}){
    		$sMethodWithoutAttributeParaTable->{"withoutParameterAndAttribute"}->{$func->id()} = 1;    		
    	}   	
    }
    
    if ($excludePrivateProtectedMethods){ 
    	my @tempMethodArray = getRefsInClass($class, "define","function ~private ~protected ~unknown ~unresolved, method ~private ~protected ~unknown ~unresolved");
   	
   	  my @pubMethodArray = ();
   	  for (my $i = 0; $i < @tempMethodArray; $i++){
   		  $pubMethodArray[$i] = $tempMethodArray[$i]->ent()->id();
   	  }
   	
   	  includeOnlyElements($sMethodWithoutAttributeParaTable, \@pubMethodArray);   	
    }

  
  
    
    #�ռ�û�б��κη������ʵ�����

    my %aTempHashTable;
    
    foreach my $key (keys %{$sAttributeReadTable}){
    	$aTempHashTable{$key} = 1;
    }

    foreach my $key (keys %{$sAttributeWriteTable}){
    	$aTempHashTable{$key} = 1;
    }

    foreach my $key (keys %{$sAttributeModifyTable}){
    	$aTempHashTable{$key} = 1;
    }

    foreach my $key (sort keys %classAttributeTable){
    	if (!$aTempHashTable{$key}){
    		$sAttributeWithoutAccessTable->{$key}->{"withoutMethodAccess"} = 1;
    	}
    }
    
    my @WithoutAccess = (keys %{$sAttributeWithoutAccessTable});    
    
    if (@WithoutAccess){
#    	print "no of attributes without access =====>", scalar @WithoutAccess, "\n";
    }
  




    my %tempInDirectCallByMethodSet = getIndirectCallByMethodSet($sDirectCallByMethodSet); 


#    print "=========Direct calls==============\n";
#	  foreach my $inKey (sort keys %{$sDirectCallByMethodSet}){
#	  	print $inKey, " calls : \n";
#	  	foreach my $calledKey (sort keys %{$sDirectCallByMethodSet->{$inKey}}){
#	  		print "\t\t", $calledKey, "\n";	  		
#	  	}
#	  }
#
#    print "=========Indirect calls==============\n";
#	  foreach my $inKey (sort keys %tempInDirectCallByMethodSet){
#	  	print $inKey, " calls : \n";
#	  	foreach my $calledKey (sort keys %{$tempInDirectCallByMethodSet{$inKey}}){
#	  		print "\t\t", $calledKey, "\n";	  		
#	  	}
#	  }
# 

	return 1;	
} # END sub buildAttributeHashTables


sub includeOnlyElements{
	#��һ��������2άhash��, �ڶ���������һ������;
	#����: ���ǵڶ��������е�����(�ַ�)�ӵ�һ��������ɾ��(��2ά�е�key)
	
	my $sHashTable = shift;
	my $sElementArray = shift;	
	
	my %bakHashTable = ();
	
	foreach my $fstKey (sort keys %{$sHashTable}){
		my %sndHashTable = %{$sHashTable->{$fstKey}};
		
		for (my $i = 0; $i < @{$sElementArray}; $i++){
			next if (!exists $sndHashTable{$sElementArray->[$i]});
			$bakHashTable{$fstKey}->{$sElementArray->[$i]} = 1;		
		}		
	}		
		
	foreach my $key (sort keys %{$sHashTable}){
		delete $sHashTable->{$key};
	}
	
	foreach my $fstKey (sort keys %bakHashTable){
		foreach my $sndKey (sort keys %{$bakHashTable{$fstKey}}){
			$sHashTable->{$fstKey}->{$sndKey} = 1;
		}
	}
	
	return;
}


sub getIndirectCallByMethodSet{
	my $sDirectCallByMethodSet = shift;
	my %InDirectCallByMethodSet = ();
	
	my %visited;
	
	foreach my $fstKey (sort keys %{$sDirectCallByMethodSet}){
		$visited{$fstKey} = 0;
		my %aMethodHashTable = %{$sDirectCallByMethodSet->{$fstKey}};
		foreach my $sndKey (sort keys %aMethodHashTable) {
			$visited{$sndKey} = 0;
		}
	}
	
	
	my $i = 0;
	foreach my $fstKey (sort keys %{$sDirectCallByMethodSet}){
		my %aMethodHashTable = %{$sDirectCallByMethodSet->{$fstKey}};
		
		my @aStack = (sort keys %aMethodHashTable);
		
		foreach my $vstKey (sort keys %visited){
			$visited{$vstKey} = 0;
		}		
		$visited{$fstKey} = 1;
				
		while (@aStack > 0){
			my $elem = shift(@aStack);
			$InDirectCallByMethodSet{$fstKey}->{$elem} = 1;
			$visited{$elem} = 1;
			
			#�������ǳ���Ҫ
			next if (!exists $sDirectCallByMethodSet->{$elem});
			
			foreach my $key (sort keys %{$sDirectCallByMethodSet->{$elem}}){
				if (!$visited{$key}){				
					push @aStack, $key;
				}
			}			
		}	
	}
		
	return %InDirectCallByMethodSet;
} # END sub getIndirectCallByMethodSet

	
sub openDatabase($)
{
    my ($dbPath) = @_;
    
    my $db = Understand::Gui::db();

    # path not allowed if opened by understand
    if ($db&&$dbPath) {
	die "database already opened by GUI, don't use -db option\n";
    }

    # open database if not already open
    if (!$db) {
	my $status;
	die usage("Error, database not specified\n\n") unless ($dbPath);
	($db,$status)=Understand::open($dbPath);
	die "Error opening database: ",$status,"\n" if $status;
    }
    return($db);
}


sub closeDatabase($)
{
    my ($db)=@_;

    # close database only if we opened it
    $db->close() if (!Understand::Gui::active());
}


sub getLastName{
	my $completeName = shift;
	
	my @wordList = split /\./, $completeName;
	
	return $wordList[$#wordList];	
}


sub getClassKey{
	my $sClass = shift;
	
	my $result = $sClass->ref()->file()->relname()."-->".getLastName($sClass->name());
	
	return $result;	
}#END sub getClassKey


sub getAncestorClasses{
	my $sClass = shift;
	my $sAncestorClassHash = shift;
	
	my $sAncestorClassLevel = {}; #����������ڵ�ǰ��Ĳ��, ֱ�Ӹ��׵Ĳ��Ϊ1, ֱ��үү�Ĳ��Ϊ2, ...
	
	#����������ļ���
	my @parentList;
	
	foreach my $parent ($sClass->refs("Base, Extend", "class", 1)){
		my $pair = {};
		$pair->{classEnt} = $parent->ent();
		$pair->{level} = 1;
		push @parentList, $pair;		
	}	
		
	while (@parentList > 0){		
		my $parentPair = shift @parentList;
		
		my $parentClassEnt = $parentPair->{classEnt};
		my $parentLevel = $parentPair->{level};
		
		my $parentClassKey = getClassKey($parentClassEnt);
		next if (exists $sAncestorClassHash->{$parentClassKey}); #��ֹ��ѭ��
		   
		$sAncestorClassHash->{$parentClassKey} = $parentClassEnt;
		$sAncestorClassLevel->{$parentLevel}->{$parentClassKey} = 1;
		
		foreach my $parent ($parentClassEnt->refs("Base, Extend", "class", 1)){
			my $pair = {};
		  $pair->{classEnt} = $parent->ent();
		  $pair->{level} = $parentLevel + 1;			
			push @parentList, $pair;
		}
	}
	
	return $sAncestorClassLevel;
}#END sub getAncestorClasses



sub getDescendentClasses{
	my $sClass = shift;
	my $sDescendentClassHash = shift;

#  print "\t\t\t computing getDescendentClasses..." if ($debug);
  	
	my @sonList;
	
	foreach my $son ($sClass->refs("Derive, Extendby", "class", 1)){
		push @sonList, $son->ent();		
	}			

	while (@sonList > 0){		
		my $currentSon = shift @sonList;
		
		my $sonClassKey = getClassKey($currentSon);
		next if (exists $sDescendentClassHash->{$sonClassKey}); #��ֹ��ѭ��
		
		$sDescendentClassHash->{$sonClassKey} = $currentSon;
		foreach my $son ($currentSon->refs("Derive, Extendby", "class", 1)){
			push @sonList, $son->ent();
		}
	}
	
#	print "....getDescendentClasses END\n" if ($debug);
	
	return 1;
}#END sub getDescendentClasses


sub getFriendClasses{
	my $sClass = shift;
	my $sFriendClassHash = shift;
	
	return if ($sClass->language() !~ m/c/i);
	
	foreach my $friend ($sClass->ents("Friend", "Class")){
		my $friendClassKey = getClassKey($friend);
		$sFriendClassHash->{$friendClassKey} = $friend;
	}
	
	return 1;
}#END sub getFriendClasses


sub getInverseFriendClasses{
	my $sClass = shift;
	my $sInverseFriendClassHash = shift;
	
	return if ($sClass->language() !~ m/c/i);
	
	foreach my $friendby ($sClass->ents("Friendby", "Class")){
		my $inverseFriendClassKey = getClassKey($friendby);
		$sInverseFriendClassHash->{$inverseFriendClassKey} = $friendby;
	}
	
	return 1;	
}#END sub getInverseFriendClasses



#�˴�otherClassHashΪ�����ࡢ�����ࡢ��Ԫ�ࡢ������Ԫ�������Ĳ��Ľ��
sub getOtherClasses{
	my $sClass = shift;
	my $sAllClassNameHash = shift;	
	my $sAncestorClassHash = shift;
	my $sDescendentClassHash = shift;	
	my $sFriendClassHash = shift;
	my $sInverseFriendClassHash = shift;
	my $sOtherClassHash = shift;
	
	my $currentClassKey = getClassKey($sClass); 
	$sOtherClassHash->{$currentClassKey} = $sClass;
	
	foreach my $classKey (keys %{$sAncestorClassHash}){		
		$sOtherClassHash->{$classKey} = $sAncestorClassHash->{$classKey};
	}
	
	foreach my $classKey (keys %{$sDescendentClassHash}){		
		$sOtherClassHash->{$classKey} = $sDescendentClassHash->{$classKey};
	}
	
	foreach my $classKey (keys %{$sFriendClassHash}){		
		$sOtherClassHash->{$classKey} = $sFriendClassHash->{$classKey};
	}

	foreach my $classKey (keys %{$sInverseFriendClassHash}){		
		$sOtherClassHash->{$classKey} = $sInverseFriendClassHash->{$classKey};
	}
	
	return 1;
}#END sub getOtherClasses


sub getAddedMethods{
#���ظ�������������ķ���(�Ǽ̳�/��override�ķ���)
  my $sClass = shift;
  my $sAddedMethodHash = shift; 
  
  my %ancestorHash;	#֮������Hash��, ���Ƕ�̳е����
  getAncestorClasses($sClass, \%ancestorHash);
	
	my %methodInAncestor; # �������еķ�����
	
	foreach my $key (keys %ancestorHash){
		my $ancestorClass = $ancestorHash{$key};
		
		my @funcList = getEntsInClass($ancestorClass, "Define", "Function ~private,Method ~private");
		
		foreach my $func (@funcList){
			my $signature = getFuncSignature($func, 1);
			$methodInAncestor{$signature} = 1;
		}
	}
	
	my @currentFuncList = getEntsInClass($sClass, "Define", "Function, Method");
	
	foreach my $func (@currentFuncList){
		my $currentSignature = getFuncSignature($func, 1);
		
		next if (exists $methodInAncestor{$currentSignature});
		
		$sAddedMethodHash->{getFuncSignature($func, 1)} = $func;
	}
	
	return 1;	
}#END sub getInheritedMethods


sub getNoOfClassAttributeInteraction{
#������c��d, ���ش�c��d����-���Խ�����Ŀ, Ҳ������c����dΪ���͵�������Ŀ
  my $sClassC = shift;
  my $sClassD = shift;
  
	my @attributeArray = $sClassC->refs("define","Member Object ~unknown ~unresolved, Member Variable ~unknown ~unresolved");
	
	my $result = 0;
  
	foreach my $attribute (@attributeArray){
		my $attributeClass = $attribute->ent()->ref("Typed", "Class");
		next if (!$attributeClass);		
		next if ($attributeClass->ent()->library() =~ m/Standard/i);
		
		my $attributeClassKey = getClassKey($attributeClass->ent());		
		next if ($attributeClassKey ne getClassKey($sClassD));		
		$result++; 
	}
	  	
	return $result;	
}#END sub getNoOfClassAttributeInteraction


sub getNoOfClassMethodInteraction{
#������c��d, ���ش�c��d����-����������Ŀ, Ҳ������c����dΪ�������ͻ��߷������͵�(�¶����)������Ŀ
  my $sClassC = shift;
  my $sClassD = shift;	
  
  my $result;
	
	my %addedMethodHash; 
	getAddedMethods($sClassC, \%addedMethodHash);
	
	foreach my $key (keys %addedMethodHash){
		my $func = $addedMethodHash{$key};
		
		my @parameters = $func->ents("Define", "Parameter");
		
		#�жϺ�����ÿ�������������ǲ�����ָ������
		foreach my $para (@parameters){			
			my $parameterClass = $para->ref("Typed", "Class");			
			next if (!$parameterClass);
			next if ($parameterClass->ent()->library() =~ m/Standard/i);
			
			my $parameterClassKey = getClassKey($parameterClass->ent());			
			next if ($parameterClassKey ne getClassKey($sClassD));					
			$result++;
		}
		
		#�жϺ����ķ��������ǲ�����ָ������
		my $returnClass = $func->ref("Typed", "Class");	
		next if (!$returnClass);
	  next if ($returnClass->ent()->library() =~ m/Standard/i);		
		
		my $returnClassKey = getClassKey($returnClass->ent());
		next if ($returnClassKey ne getClassKey($sClassD));
		$result++;
	}
	
	return $result;	
}#END sub getNoOfClassMethodInteraction


sub getClassnameFromTypename{
#���������õ�����, ��ȥ��*, const, &�ȷ���
  my $sTypename = shift;
  
  
	$sTypename =~ s/\*//g;
	$sTypename =~ s/&//g;
	$sTypename =~ s/const//g;
	$sTypename =~	s/^\s+//;
	$sTypename =~ s/\s+$//;
	
	return $sTypename;	
}#END sub getClassnameFromTypename



sub getNoOfMethodMethodInteraction{
#������c��d, ���ش�c��d�ķ���-����������Ŀ, Ҳ������c�е���Щ��������d�еķ���������
#��d�з���Ϊ�����ķ�����Ŀ	
  my $sClassC = shift;
  my $sClassD = shift;	

	my @methodArray = $sClassC->refs("define", "function ~unresolved ~unknown, method ~unresolved ~unknown");    
	
	my $result = 0;
  
  #ͳ����$sClassC�е�����$sClassD�з����ķ�����Ŀ
  foreach my $method (@methodArray){
  	my @calledFuncSet = $method->ent()->refs("call", "function ~unresolved ~unknown, method ~unresolved ~unknown");
  	foreach my $func (@calledFuncSet){
  		my $calledClass = $func->ent()->ref("Definein", "Class");
  		next if (!$calledClass);  		
  		next if ($calledClass->ent()->library() =~ m/Standard/i);
  		
  		my $calledClassName = getLastName($calledClass->ent()->name());
  		
  		next if ($calledClassName ne getLastName($sClassD->name())); 

  		$result++;
  	}
  }
  
  #ͳ����$sClassD�з���Ϊ������(��$sClassC)������Ŀ
  
  
  return $result;		
}#END sub getNoOfMethodMethodInteraction



sub IMC{
	#���㷽�����ڲ�������, ʵ������Halstead��effort��ʽ.  ���ý���ṩ���ӳ���CBI, ��Degree of coupling of inheritance in a class.
	#Ref: E.M. KIM, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity metrics. COMPSAC 1996.
	my $sEnt = shift;   # ������ʵ��
	
  
 	my $func = $sEnt;
  
  my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);	  
  return 0 if ($lexer eq "undef");
    
	my ($n1, $n2, $N1, $N2) = scanEntity($lexer,$startLine,$endLine);
	  
  # do the calcs
  my ($n, $N) = ($n1 + $n2, $N1 + $N2);
  
#  print "\t n1 = ", $n1, "\n";
#  print "\t n2 = ", $n2, "\n";
#  print "\t N1 = ", $N1, "\n";
#  print "\t N2 = ", $N2, "\n";
	 	
	return 0 if ($n1 < 1);
	return 0 if ($n2 < 1);
	return 0 if ($N2 < 1); 	
	 	
  my $result = $N * ((log $n)/(log 2)) / ($n1/2 * $n2/$N2);

  return $result;
} # END sub IMC



sub getNoReserveWords{
	#���㷽�����еı�������Ŀ, �ṩ���ӳ���MPCnew
	my $sEnt = shift;
	
	my $func = $sEnt;
	
	my %reserveWordsHash = ( if => 1,
	                         else => 1,  
		                       switch => 1,
		                       case => 1,
		                       default => 1,
		                       for => 1,
		                       while => 1,
		                       do => 1,
		                       repeat => 1,
		                       until => 1,
		                       next => 1,
		                       continue => 1,
		                       break => 1,
		                       throw => 1,
		                       try => 1,
		                       catch => 1
		                      );
	
  my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($func);	  
  return 0 if ($lexer eq "undef");
  
  my $result = 0;
	    
  foreach my $lexeme ($lexer->lexemes($startLine,$endLine)) 
  {
     next if ($lexeme->token() ne "Keyword");  
     next if (!exists $reserveWordsHash{$lexeme->text()});     
     $result++;
  }	
	
	return $result;
}#END sub getNoReserveWords


sub PIM{
	#���ض�̬���õķ�����
	#Ŀǰֻ�ܷ��ؾ�̬���õķ�����, �Ժ�����ɶ�̬���õķ�����  
	my $sEnt = shift;
	my $polyCalledMethodSet = shift; #��̬���õķ�������, ��һ��hash��. key������+����������, value��(funcEnt, callCount)��
	
#	print "\n\t\t\t computing PIM..." if ($debug);	
	
	my $callingFunc = $sEnt;  
	
	my @calledMethodSet = $callingFunc->refs("Call", "function ~unknown ~unresolved, method ~unknown ~unresolved");
	
	foreach my $calledMethod (@calledMethodSet){
		my $calledFunc = $calledMethod->ent();
		my $calledClass = $calledFunc->ref("Definein", "Class ~unknown ~unresovled");
  	next if (!$calledClass);  		
  	next if ($calledClass->ent()->library() =~ m/Standard/i);  		
  	
  	my $calledFuncSignature = getFuncSignature($calledFunc,1);	
  	
  	my $funcKey = getLastName($calledClass->ent()->name())."::".$calledFuncSignature;
  	
  	if (!exists $polyCalledMethodSet->{$funcKey}){
  		$polyCalledMethodSet->{$funcKey}->{funcEnt} = $calledFunc;
  		$polyCalledMethodSet->{$funcKey}->{callCount} = 1;
  	}
  	else{
  		$polyCalledMethodSet->{$funcKey}->{callCount}++;
  	}
    
    my $callingClass = $callingFunc->ref("Definein", "Class ~unknown ~unresovled");
    my $callingFuncSignature = getFuncSignature($callingFunc,1);	
    my $callingFuncKey = getLastName($callingClass->ent()->name())."::".$callingFuncSignature;
    
#    print "\t\t", $callingFuncKey, "====>", $funcKey, "\n";
##����Ĵ��뷵�صļ��ϱ������Ķ�̬���ü�Ҫ��ܶ�, �д����  	
##  	print "\t\t called func = ", $calledFunc->name(), "\n"; 
##  	print "\t\t called class = ", $calledClass->ent()->name(), "\n";
#    my %descendentClassHash; 
#    
#    getDescendentClasses($calledClass->ent(), \%descendentClassHash);
#  	
##  	print "NO descendent class ====> ", scalar (keys %descendentClassHash), "\n";
#  	
#  	foreach my $key (keys %descendentClassHash){
#  		my $currentClass = $descendentClassHash{$key};
#  		
# # 		print "\t\t currentClass = ", $currentClass->name(),"\n";
#  		
#  		my $calledFuncEnt; 
#  		my $find;
#  		($find, $calledFuncEnt) = IsImplementedInClass($currentClass, $calledFuncSignature);
#  		next if (!$find);
#  		  		
#  #		print "=========\n";
#
#      my $funcKey = getLastName($currentClass->name())."::".$calledFuncSignature;   		
#      
#  	  if (!exists $polyCalledMethodSet->{$funcKey}){
#  		  $polyCalledMethodSet->{$funcKey}->{funcEnt} = $calledFuncEnt;
#  		  $polyCalledMethodSet->{$funcKey}->{callCount} = 1;
#  	  }
#  	  else{
#  		  $polyCalledMethodSet->{$funcKey}->{callCount}++;
#  	  }      
#  		
##  		print "\t\t class name = ", $currentClass->name(), "\n";
##  		print "\t\t funcEnt = ", $calledFuncEnt->name(), "\n";
#  	}
  }
  
#	print ".....PIM END\n" if ($debug);
	
}#END sub PIM


sub IsImplementedInClass{
	#����һ����, �Լ�һ�������Ļ���, �жϸ÷������ɸ��ඨ��
	my $sClass = shift;
	my $sMethodSignature = shift;
	my $sCalledFuncEnt; #������ڶ���, �򷵻ظú���ʵ��
	
	my @methodArray = getEntsInClass($sClass, "define", "function ~unknown ~unresolved, method ~unknown ~unresolved");
	
	my $find = 0;
	
	my $i = 0;
	while (!$find && $i < @methodArray){
		my $key = getFuncSignature($methodArray[$i], 1);		
		if ($sMethodSignature eq $key){
			$find = 1;
			$sCalledFuncEnt = $methodArray[$i];
		}
		$i++;
	}
	
	return wantarray?(0, 0):0 if (!$find);	
	
	return wantarray?(1, $sCalledFuncEnt):1;
}#END sub IsImplementedInClass



sub getLexerStartAndEndLine{
	my $sEnt = shift;
	
	my $lexer = $sEnt->ref()->file()->lexer();
		
  my ($startRef) = $sEnt->refs("definein","",1);
  ($startRef) = $sEnt->refs("declarein","",1) unless ($startRef);  
	my ($endRef) = $sEnt->refs("end","",1);
	
	return ("undef", 0, 0) unless ($startRef and $endRef);
		
	my $startLine = $startRef->line();
	my $endLine = $endRef->line();
		
	my $test = $lexer->lexemes($startLine, $endLine);
	while (!$test and $endLine > $startLine){
		$endLine = $endLine - 1;
		$test = $lexer->lexemes($startLine, $endLine);
	}		
	
	return ($lexer, $startLine, $endLine);
}


sub IsMethodInClassHeader{
	my $sClass = shift;
	my $sFunc = shift;
	
	return 1;
	
#	print "sClass->ref()->file()->name() = ", $sClass->ref()->file()->name(), "\n";
#	print "sFunc->ref()->file()->name()) = ", $sFunc->ref()->file()->name(), "\n";

	return 1 if ($sClass->ref()->file()->relname() ne $sFunc->ref()->file()->relname()
	             && $sClass->ref()->file()->name() eq $sFunc->ref()->file()->name());

	return 0 if ($sClass->ref()->file()->name() ne $sFunc->ref()->file()->name());
	
  my ($classStartRef) = $sClass->refs("definein", "", 1);
	my ($classEndRef) = $sClass->refs("end","",1);
#	
#	my @refs = $sClass->refs();
#	print "@refs = ", scalar @refs, "\n";
#	foreach my $sref (@refs){
#		print "ref name = ", $sref->ent()->name(), "\n";
#		print "ref line = ", $sref->line(), "\n";
#		
#	}
		
	my $classStartLine = $classStartRef->line();
	my $classEndLine = $classEndRef->line();    
	
#	print "classStartLine = ", $classStartLine, "\n";
#	print "classEndLine = ", $classEndLine, "\n";

  my ($funcStartRef) = $sFunc->refs("definein", "", 1);
	my ($funcEndRef) = $sFunc->refs("end","",1);
		
	my $funcStartLine = $funcStartRef->line();
	my $funcEndLine = $funcEndRef->line();    
#
#	print "funcStartLine = ", $funcStartLine, "\n";
#	print "funcEndLine = ", $funcEndLine, "\n";

	
	return 1 if ($funcStartLine >= $classStartLine && $funcEndLine <= $classEndLine);
	
	return 0;	
}


sub getTimeInSecond{
	my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
	
	my $timePoint = $hour*3600 + $min * 60 + $sec;
	
	return $timePoint;
}

sub reportComputeTime{
	my $startTime = shift; #����Ϊ��λ
	my $subProgramName = shift; #ͳ�Ƹ��ӳ����ʱ��

  my $endTime = getTimeInSecond();
	
	my $timeInComputation = $endTime - $startTime;
	
	print "\t\t\t ", $subProgramName, " takes ", $timeInComputation, " seconds \n"; 
}


sub getEntsInClass{
	#�������ж�������Ի��߷���ʵ��
	#��Ҫ����,һ����Ŀ�о��ж���汾����, ��ǰ��understand���ݿ�������, ����ͬ��(������ͬ������)�����Ի��߷���
	#�ϲ���һ��. ���, Ҫ��ȷ�����������
	my $sClass = shift;
	my $refKindString = shift;
	my $entKindString = shift;
	
	my @entArray = $sClass->ents($refKindString, $entKindString);
	
	return @entArray;
	
	
#	#�ඨ�����ڵ��ļ��������
#	my $fileRelNameOfClass = $sClass->ref()->file()->relname();	
#	#�ඨ�����ڵ��ļ���
#	my $fileNameOfClass = $sClass->ref()->file()->name();
#	
#	my @result = ();
#	
#	foreach my $ent (@entArray){		
#		#ʵ�嶨�����ڵ��ļ��������
#		my $fileRelNameOfEnt = $ent->ref()->file()->relname();
#		#ʵ�嶨�����ڵ��ļ���
#		my $fileNameOfEnt = $ent->ref()->file()->name();
#		
#		#����������ͬ, ������ͬ, ������ͬ�汾���ļ�
#		next if ($fileRelNameOfClass ne $fileRelNameOfEnt  
#		         && $fileNameOfClass eq $fileNameOfEnt);
#		
#		push @result, $ent;		
#	}
#	
#	return @result;	
}#END sub getEntsInClass


sub getRefsInClass{
	#�������ж�������Ի��߷���ʵ��
	#��Ҫ����,һ����Ŀ�о��ж���汾����, ��ǰ��understand���ݿ�������, ����ͬ��(������ͬ������)�����Ի��߷���
	#�ϲ���һ��. ���, Ҫ��ȷ�����������
	my $sClass = shift;
	my $refKindString = shift;
	my $entKindString = shift;
	
	my @entArray = $sClass->refs($refKindString, $entKindString);
	
	return @entArray;
	
#	#�ඨ�����ڵ��ļ��������
#	my $fileRelNameOfClass = $sClass->ref()->file()->relname();	
#	#�ඨ�����ڵ��ļ���
#	my $fileNameOfClass = $sClass->ref()->file()->name();
#	
#	my @result = ();
#	
#	foreach my $aref (@entArray){				
#		my $ent = $aref->ent();
#		#ʵ�嶨�����ڵ��ļ��������
#		my $fileRelNameOfEnt = $ent->ref()->file()->relname();
#		#ʵ�嶨�����ڵ��ļ���
#		my $fileNameOfEnt = $ent->ref()->file()->name();
#		
#		#����������ͬ, ������ͬ, ������ͬ�汾���ļ�
#		next if ($fileRelNameOfClass ne $fileRelNameOfEnt  
#		         && $fileNameOfClass eq $fileNameOfEnt);
#		
#		push @result, $aref;		
#	}
#	
#	return @result;	
}#END sub getRefsInClass



############################################################################
###�Զ���SLOC#####ע�����´���changeproness�ļ���������Ѿ�����, ���������#
############################################################################

sub SLOC{
	my $sClass = shift;
	
	my $noOfHeaderStatements = getLinesOfCode($sClass);

  	my $noOfMethodStatements = 0;
 	my @methodArray = getRefsInClass($sClass, "define","function, method ~unknown ~unresolved");
     	
  	foreach my $method (@methodArray){
  		my $func = $method->ent();     		
    
    		#���������Ķ����������ͷ�Ķ�����,������(����Ҫ, �����ظ�����)
    		next if IsMethodInClassHeader($sClass, $func);     		
    
    		$noOfMethodStatements = $noOfMethodStatements + getLinesOfCode($func);
  	}
	  	
	#�Լ�����SLOC (Understand�ļ��㷽���г��������еı���ָʾ���������)
	my $result = $noOfHeaderStatements + $noOfMethodStatements;
	
	return $result;  	
}#END sub SLOC


sub getLinesOfCode
{
    my $sEnt = shift;

    my $result;

    # create lexer object
    my ($lexer,$status) = $sEnt->ref()->file()->lexer();
    die "\ncan't open lexer on class, error: $status" if ($status);

    # scan lexemes
    # slurp lines with comments extracted, omit empty lines
    my $text;
    
    my ($lexer, $startLine, $endLine) = getLexerStartAndEndLine($sEnt); 
		return 0 if ($lexer eq "undef");
		
#		print "name = ", $sEnt->longname(), "\n";
#		print "startLine = ", $startLine, "\n";
#		print "endLine = ", $endLine, "\n";
    
    
    foreach my $lexeme ($lexer->lexemes($startLine, $endLine)){
     	# save non-blank strings when newline is encountered
	    if ( $lexeme->token() eq "Newline" ) {
	      if ( $text =~ /[0-9a-zA-Z_{}]/ ){	      	
		       $result++;
	      }

	      # clear text
	      $text = "";
	      next;
	    }

	    # append to text if code, skipping all comments
	    if ( $lexeme->token() !~ /Comment/g ) {
	       $text .= $lexeme->text();	    
	    } 

    } #End for

    return $result;
}#END sub getLinesOfCode













#NOP:   CountClassBase
#CBO:   CountClassCoupled
#NOC:   CountClassDerived
#NIM:   CountDeclInstanceMethod
#NIV:   CountDeclInstanceVariable
#WMC:   CountDeclMethod
#RFC:   CountDeclMethodAll
#DIT:   MaxInheritanceTree
#LCOM:  PercentLackOfCohesion

#
#
#
#=====================���ݼ���������������˵��===================================
#(�޶�����:2008��7��23��)
#
#1.�����Զ���
#  (1)CDE 
#     a. ȫ��: Class Definition Entropy
#     b. ����: J. Bansiya, C. Davis, L. Etzkorn. An entropy-based complexity measure for object-oriented designs. Theory 
#              and Practice of Object Systems, 5(2), 1999: 111-118.
#     c. ����: 
#     
#  (2)CIE    
#     a. ȫ��: Class Implementation Entropy
#     b. ����: ͬCIE
#     c. ����:    
#
#  (3)WMC    
#     a. ȫ��: Weighted Method Per Class 
#     b. ����: S.R. Chidamber, C.F. Kemerer.A metrics suite for object-oriented design. IEEE TSE, 20(6), 1994: 476-493.
#     c. ����: 
#     
#  (4)SDMC    
#     a. ȫ��: Standard Deviation Method Complexity 
#     b. ����: Michura J, Capretz MAM. Metrics suite for class complexity. International Conference on Information Technology: 
#              Coding and Computing, 2005; 404�C409.
#     c. ����: 
#
#  (5)AWMC    
#     a. ȫ��: Average Method Complexity (the average of cyclomatic complexity of all methods in a class, i.e. CCAvg)
#     b. ����: Etzkorn LH, Bansiya J, Davis C. Design and code complexity metrics for OO classes. Journal of Object-oriented
#              Programming 1999; 12(1):35�C40.
#     c. ����: 
#
#  (6)CCMax    
#     a. ȫ��: Maximum cyclomatic complexity of a single method of a class 
#     b. ����: H.M. Olague, L.H. Etzkorn, S.L. Messimer, H.S. Delugach. An empirical validation of object-oriented class
#              complexity metrics and their ability to predict error-prone classes in highly iterative, or agile, software:
#              a case study. Journal of Software Maintenance and Evolution: Research and Practice, 2008, 3.
#     c. ����: 
#
#  (7)NTM    
#     a. ȫ��: Number of Trival Methods (the number of local methods in the class whose McCabe complexity value is 
#              equal to one.
#     b. ����: McCabe TJ. A complexity measure. IEEE Transactions on Software Engineering 1976; 2(4):308�C320.
#     c. ����: 
#
#  (8)CC1    
#     a. ȫ��: Class Complexity One 
#     b. ����: Y.S. Lee, B.S. Liang, F.J. Wang. Some complexity metrics for OO programs based on information flow. IEEE 
#              COMPEURO 1993: 302-310.
#     c. ����: 
#     
#  (9)CC2    
#     a. ȫ��: Class Complexity Two 
#     b. ����: ͬCC1
#     c. ����: 
#     
#  (10)CC3    
#     a. ȫ��: Class Complexity Three 
#     b. ����: K. Kim, Y. Shin, C. Wu. Complexity measures for OO program based on the entropy. APSEC 1995: 127-136.
#     c. ����: 
#
#
#2.����Զ���
#  (1)CBO    
#     a. ȫ��: Coupling Between Object 
#     b. ����: ͬWMC
#     c. ����: 
#
#  (2)RFC    
#     a. ȫ��: Response For a Class, ���������ķ���������ֱ�ӻ��߼�ӵ��õķ��� 
#     b. ����: 
#     c. ����:      
#
#  (3)RFC1    
#     a. ȫ��: Response For a Class, ֻ����������ķ�����ֱ�ӵ��õķ��� 
#     b. ����: S.R. Chidamber, C.F. Kemerer. Towards a metrics suite for object-oriented design. OOPSLA 1991: 197-211.
#     c. ����:      
#
#  (3)MPC
#     a. ȫ��: Message Passing Coupling
#     b. ����: W. Li, S. Henry. Object-oriented metrics that predict maintainability, JSS, 23(2), 1993: 11-122
#     c. ����:   
#
#  (4)MPCNew
#     a. ȫ��: Message Passing Coupling (Number of send statements in a class)
#     b. ����: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity 
#              metrics. COMPSAC 1996: 104-109.  
#     c. ����:   
#
#  (5)DAC
#     a. ȫ��: Data Abstraction Coupling: �������������������Ŀ
#     b. ����: ͬMPC
#     c. ����:   
#
#  (6)DACquote
#     a. ȫ��: Data Abstraction Coupling: ������������������Ŀ
#     b. ����: ͬMPC
#     c. ����:   
#
#  (7)ICP
#     a. ȫ��: Information-flow-based Coupling
#     b. ����: Y.S. Lee, B.S. Liang, S.F. Wu, F.J. Wang. Measuring the coupling and cohesion of an object-oriented 
#              program based on information flow. ICSQ 1995.
#     c. ����:   
#
#  (8)IHICP
#     a. ȫ��: Information-flow-based inheritance Coupling
#     b. ����: ͬICP
#     c. ����: 
#
#  (9)NIHICP
#     a. ȫ��: Information-flow-based non-inheritance Coupling
#     b. ����: ͬICP
#     c. ����: 
#     
#  (10)IFCAIC
#     a. ȫ��: Inverse friends class-attribute interaction import coupling
#     b. ����: L.C. Briand, P. Devanbu, W. Melo. An investigation into coupling metrics for C++. ICSE 1997: 412-421.
#     c. ����: 
#     
#  (11)ACAIC
#     a. ȫ��: Ancestor classes class-attribute interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (12)OCAIC
#     a. ȫ��: Others class-attribute interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#     
#  (13)FCAEC
#     a. ȫ��: Friends class-attribute interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:     
#
#  (14)DCAEC
#     a. ȫ��: Descendents class class-attribute interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:   
#
#  (15)OCAEC
#     a. ȫ��: Others class-attribute interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:   
#
#  (16)IFCMIC
#     a. ȫ��: Inverse friends class-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (17)ACMIC
#     a. ȫ��: Ancestor class class-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#     
#  (18)OCMIC
#     a. ȫ��: Others class-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����:      
#     
#  (19)FCMEC
#     a. ȫ��: Friends class-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (20)DCMEC
#     a. ȫ��: Descendents class-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (21)OCMEC
#     a. ȫ��: Others class-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (22)OMMIC
#     a. ȫ��: Others method-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����: 
#
#  (23)IFMMIC
#     a. ȫ��: Inverse friends method-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����:
#
#  (24)AMMIC
#     a. ȫ��: Ancestor class method-method interaction import coupling
#     b. ����: ͬIFCAIC
#     c. ����:
#
#  (25)OMMEC
#     a. ȫ��: Others method-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:
#
#  (26)FMMEC
#     a. ȫ��: Friends method-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:
#     
#  (27)DMMEC
#     a. ȫ��: Descendents method-method interaction export coupling
#     b. ����: ͬIFCAIC
#     c. ����:    
#
#  (28)CBI
#     a. ȫ��: Degree of coupling of inheritance
#     b. ����: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity 
#              metrics. COMPSAC 1996: 104-109.  
#     c. ����: 
#
#  (29)UCL
#     a. ȫ��: Number of classes used in a class except for ancestors and children
#     b. ����: ͬCBI  
#     c. ����:  
#
#  (30)CC
#     a. ȫ��: Class Coupling
#     b. ����: C. Rajaraman, M.R. Lyu. Reliability and maintainability related software coupling metrics in C++ programs.
#     c. ����:  
# 
#  (31)AMC
#     a. ȫ��: Average Method Coupling
#     b. ����: ͬCC
#     c. ����:  
#  
#
#3.�̳�����ض���
#  (1)NOC
#     a. ȫ��: Number Of Child Classes
#     b. ����: ͬWMC
#     c. ����:  
#
#  (2)NOP
#     a. ȫ��: Number Of Parent Classes
#     b. ����: M. Lorenz, J. Kidd. Object-oriented software metrics: a practical guide. Prentice-Hall, 1994
#     c. ����:  
#
#  (3)DIT
#     a. ȫ��: Depth of Inheritance Tree
#     b. ����: ͬWMC
#     c. ����: 
#     
#  (4)AID
#     a. ȫ��: Average Inheritance Depth of a class
#     b. ����: B.Henderson-sellers. Object-oriented metrics: measures of complexity, Prentice Hall, 1996
#     c. ����:      
#
#  (5)CLD
#     a. ȫ��: Class-to-Leaf Depth
#     b. ����: D.P. Tegarden, S.D. Sheetz, D.E. Monarchi. A software complexity model of object-oriented systems. 
#              Decision SupportSystems, 13(3�C4), 1995: 241�C262.
#     c. ����:      
#
#  (6)NOD
#     a. ȫ��: Number Of Descendents
#     b. ����: ͬCLD
#     c. ����: 
#
#  (7)NOA
#     a. ȫ��: Number Of Ancestors
#     b. ����: ͬCLD
#     c. ����: 
#
#  (8)NMO
#     a. ȫ��: Number of Methods Overridden
#     b. ����: ͬNOP
#     c. ����: 
#
#  (9)NMI
#     a. ȫ��: Number of Methods Inherited
#     b. ����: ͬNOP
#     c. ����: 
#
#  (10)NMA
#     a. ȫ��: Number Of Methods Added
#     b. ����: ͬNOP
#     c. ����: 
#
#  (11)SIX
#     a. ȫ��: Specialization IndeX   =  NMO * DIT / (NMO + NMA + NMI)
#     b. ����: ͬNOP
#     c. ����: 
#     
#  (12)PII
#     a. ȫ��: Pure Inheritance Index
#     b. ����: B.K. Miller, P. Hsia, C. Kung. Object-oriented architecture measures. 32rd Hawaii International Conference 
#              on System Sciences 1999 
#     c. ����: 
#     
#  (13)SPA
#     a. ȫ��: static polymorphism in ancestors
#     b. ����: S. Benlarbi, W.L. Melo. Polymorphism measures for early risk prediction. 
#              ICSE 1999: 334-344.
#     c. ����:
#
#  (14)SPD
#     a. ȫ��: static polymorphism in decendants
#     b. ����: ͬSPA
#     c. ����:
#     
#  (15)DPA
#     a. ȫ��: dynamic polymorphism in ancestors
#     b. ����: ͬSPA
#     c. ����:     
#
#  (16)DPD
#     a. ȫ��: dynamic polymorphism in decendants
#     b. ����: ͬSPA
#     c. ����: 
#     
#  (17)SP
#     a. ȫ��: static polymorphism in inheritance relations
#     b. ����: ͬSPA
#     c. ����: 
#     
#  (18)DP
#     a. ȫ��: dynamic polymorphism in inheritance relations
#     b. ����: ͬSPA
#     c. ����:    
#     
#  (19)CHM
#     a. ȫ��: Class hierarchy metric
#     b. ����: J.Y. Chen, J.F. Lu. A new metric for OO design. IST, 35(4): 1993.
#     c. ����:  
#     
#  (20)DOR
#     a. ȫ��: Degree of reuse by inheritance
#     b. ����: E.M. Kim, S. Kusumoto, T. Kikuno. Heuristics for computing attribute values of C++ program complexity 
#              metrics. COMPSAC 1996: 104-109.
#     c. ����:             
#     
#
#4.��ģ����
#  (1)NMIMP
#     a. ȫ��: Number Of Methods Implemented in a class
#     b. ����: 
#     c. ����:    
#
#  (2)NAIMP
#     a. ȫ��: Number Of Attributes Implemented in a class
#     b. ����: 
#     c. ����: 
#  
#  (3)SLOC
#     a. ȫ��: source lines of code
#     b. ����: 
#     c. ����:
#  
#  (4)SLOCExe
#     a. ȫ��: source lines of executable code
#     b. ����: 
#     c. ����:
#
#  (5)stms
#     a. ȫ��: number of statements
#     b. ����: ͬNM
#     c. ����:
#
#  (6)stmsExe
#     a. ȫ��: number of executable statements
#     b. ����: 
#     c. ����:
#     
#  (7)NM
#     a. ȫ��: number of all methods (inherited, overriding, and non-inherited) methods of a class
#     b. ����: L.C. Briand, J. Wust, J.W. Daly, D.V. Porter. Exploring the relationships between design measures and
#              software quality in object-oriented systems. JSS, 51(3), 2000: 245-273.
#     c. ����:     
#     
#  (8)Nmpub
#     a. ȫ��: number of public methods implemented in a class
#     b. ����: ͬNM
#     c. ����:     
#     
#  (9)NMNpub
#     a. ȫ��: number of non-public methods implemented in a class
#     b. ����: ͬNM
#     c. ����:     
#
#  (10)NumPara
#     a. ȫ��: sum of the number of parameters of the methods implemented in a class
#     b. ����: ͬNM
#     c. ����:     
#
#  (11)NIM    
#     a. ȫ��: Number of Instance Methods (the number in an instance object of a class. This is different from a class
#              method, which refers to a method which only operates on data belonging to the class itself, not on data 
#              that belong to individual objects. 
#     b. ����: Lorenz M, Kidd J. Object-oriented Software Metrics, 1994; 146.
#     c. ����: 
#     
#  (12)NCM    
#     a. ȫ��: Number of Class Methods
#     b. ����: Lorenz M, Kidd J. Object-oriented Software Metrics, 1994; 146.
#     c. ����: 
#     
#  (13)NLM    
#     a. ȫ��: Number of Local Methods (NLM = NIM + NCM = NMIMP) 
#     b. ����: 
#     c. ����: 
#     
#  (14)AvgSLOC    
#     a. ȫ��: Average Source Lines of Code (Average of the lines of code of a class) 
#     b. ����: H.M. Olague, L.H. Etzkorn, S.L. Messimer, H.S. Delugach. An empirical validation of object-oriented class
#              complexity metrics and their ability to predict error-prone classes in highly iterative, or agile, software:
#              a case study. Journal of Software Maintenance and Evolution: Research and Practice, 2008, 3.
#     c. ����: 
#
#  (15)AvgSLOCExe    
#     a. ȫ��: Average Source Lines of Executable Code (Average of teh executable lines of code of a class) 
#     b. ����: H.M. Olague, L.H. Etzkorn, S.L. Messimer, H.S. Delugach. An empirical validation of object-oriented class
#              complexity metrics and their ability to predict error-prone classes in highly iterative, or agile, software:
#              a case study. Journal of Software Maintenance and Evolution: Research and Practice, 2008, 3.
#     c. ����: 
#
#
#5.�ھ��Զ���
#  (1)LCOM1
#     a. ȫ��: 
#     b. ����: ͬRFC1
#     c. ����:
#
#  (2)LCOM2
#     a. ȫ��: 
#     b. ����: ͬWMC
#     c. ����:
#
#  (3)LCOM3
#     a. ȫ��: 
#     b. ����: ͬCo
#     c. ����:
#
#  (4)LCOM4
#     a. ȫ��: 
#     b. ����: ͬCo
#     c. ����:
#
#  (5)Co
#     a. ȫ��: 
#     b. ����: M. Hitz, B. Montazeri. Measuring coupling and cohesion in object-oriented systems. SAC 1995: 25-27
#     c. ����:
#
#  (6)NewCo
#     a. ȫ��: 
#     b. ����: ͬNewLCOM5
#     c. ����:
#
#  (7)LCOM5
#     a. ȫ��: 
#     b. ����: ͬAID
#     c. ����:
#
#  (8)NewLCOM5 
#     a. ȫ��: also called NewCoh/Coh
#     b. ����: L.C. Briand, J.W. Daly, J. Wust. A unified framework for cohesion measurement in object oriented systems.
#              Empirical Software Engineering, 3(1), 1998: 65-117.
#     c. ����:
#
#  (9)LCOM6
#     a. ȫ��: based on parameter names.
#     b. ����: J.Y. Chen, J.F. Lu. A new metric for OO design. IST, 35(4): 1993.
#     c. ����:     
#     
#  (10)LCC 
#     a. ȫ��: Loose Class Cohesion
#     b. ����: J.M. Bieman, B.K. Kang. Cohesion and reuse in an object-oriented system. Proceedings of ACM Symposium on 
#              Software Reusability, 1995: 259-262.
#     c. ����:
#     
#  (11)TCC 
#     a. ȫ��: Tight Class Cohesion
#     b. ����: ͬLCC
#     c. ����:     
#
#  (12)ICH 
#     a. ȫ��: Information-flow-based Cohesion
#     b. ����: ͬICP
#     c. ����:
# 
#  (13)DCd 
#     a. ȫ��: Degree of Cohesion based Direct relations between the public methods
#     b. ����: Linda Badri and Mourad Badri. A Proposal of a New Class Cohesion Criterion: An Empirical Study.
#              Journal of Object Technology, 3(4), 2004: 145-159.
#     c. ����:
#
#  (14)DCi 
#     a. ȫ��: Degree of Cohesion based Indirect relations between the public methods
#     b. ����: ͬDCd
#     c. ����:
# 
#  (15)CBMC 
#     a. ȫ��: 
#     b. ����: 
#     c. ����:
#
#  (16)ICBMC 
#     a. ȫ��: 
#     b. ����: 
#     c. ����:
#
#  (17)ACBMC 
#     a. ȫ��: 
#     b. ����: 
#     c. ����:
#  
#  (18)C3 
#     a. ȫ��: conceptual cohesion of classes
#     b. ����: A. Marcus, D. Poshyvanyk. The conceptual cohesion of classes. ICSM 2005.
#     c. ����:
#  
#  (19)LCSM 
#     a. ȫ��: Lack of Conceptual similarity between Methods
#     b. ����: ͬC3
#     c. ����:
#
#  (20)OCC 
#     a. ȫ��: Opitimistic Class Cohesion
#     b. ����: Aman H., Yamasaki K., Yamada H. Noda M., ��A Proposal of Class Cohesion Metrics Using Sizes of Cohesive 
#              Parts��, Knowledge-based Software Engineering, T. welzer et al.(Eds.), pp102-107, IOS Press, Sept. 2002.
#     c. ����:
#
#  (21)PCC 
#     a. ȫ��: Pessimistic Class Cohesion
#     b. ����: ͬOCC
#     c. ����:
#
#  (22)CAMC 
#     a. ȫ��: Cohesion Among Methods in a Class
#     b. ����: J. Bansiya, L. Etzkorn, C. Davis, W. Li. A class cohesion metric for object-oriented designs. 
#              JOOP, 11(8), 1999: 47-52.
#     c. ����:
#
#  (23)iCAMC 
#     a. ȫ��: ������������ֵ���͵�CAMC
#     b. ����: 
#     c. ����:
#
#  (24)CAMCs 
#     a. ȫ��: ����self���͵�CAMC
#     b. ����: ͬCAMC
#     c. ����:     
#     
#  (25)iCAMCs 
#     a. ȫ��: ������������ֵ���ͺ�self���͵�CAMC
#     b. ����: 
#     c. ����:     
#
#  (26)NHD 
#     a. ȫ��: Normalized Hamming Distance metric
#     b. ����: S. Counsell, E. Mendes, S. Swift, A. Tucker. Evaluation of an object-oriented cohesion
#             metric through Hamming distances. Tech. Rep. BBKCS-02-10, Birkbeck College, University of London, UK, 2002.
#     c. ����: 
#
#  (27)iNHD 
#     a. ȫ��: 
#     b. ����: 
#     c. ����: 
#     
#  (28)NHDs 
#     a. ȫ��: 
#     b. ����: 
#     c. ����: 
#     
#  (29)iNHDs 
#     a. ȫ��: 
#     b. ����: 
#     c. ����: 
#
#  (30)SNHD 
#     a. ȫ��: Scaled NHD metric 
#     b. ����: S. Counsell, S. Swift, J. Crampton. The interpretation and utility 
#              of three cohesion metrics for object-oriented design. 
#              ACM Transactions on Software Engineering and Methodology, 15(2), 2006: 123-149.
#     c. ����:     
#     
#  (31)iSNHD 
#     a. ȫ��:  
#     b. ����: 
#     c. ����:     
#     
#  (32)SCOM 
#     a. ȫ��: Sensitive Class Cohesion Metric 
#     b. ����: L. Fernandez, R. Pena. A sensitive metric of class cohesion. International Journal of 
#              Information Theories & Applications, 13(1), 2006: 82-91.
#     c. ����:     
#     
#  (33)CAC 
#     a. ȫ��: Class Abstraction Cohesion 
#     b. ����: B.K. Miller, P. Hsia, C. Kung. Object-oriented architecture measures. 
#              32rd Hawaii International Conference on System Sciences 1999
#     c. ����:     
#     
#     
#6.��������
#  (1)OVO 
#     a. ȫ��: parametric overloading metric
#     b. ����: ͬSPA
#     c. ����:   
#     
#  (2)MI 
#     a. ȫ��:  Maintainability Index
#     b. ����: 
#     c. ����:   
#     
#     
#7. �ײ��Զ���
#   (1) testingCSLOC
#     a. ȫ��: �������SLOC
#     b. ����: M. Bruntink, A. van Deursen. An empirical study into class testability. JSS, 79(9), 2006: 1219-1232.
#   
#   (2) countOfAssertFunction
#     a. ȫ��: ��������Assert������Ŀ
#     b. ����: ͬtestingCSLOC
#
#8. Change-proneness����
#   (1) totalChangedsloc
#     a. ȫ��: ����������ɾ����SLOC֮��(һ��changed����俴��һ��ɾ����һ������, ��˼�2)
#     b. ����: (I)E. Arisholm, L.C. Briand, A. Foyen. Dynamic coupling measurement for object-oriented software. 
#                 IEEE TSE, 30(8), 2004: 491-506.
#              (II) W. Li, S.M. Henry. Object-oriented metrics that predict maintainability. JSS, 23(2), 1993: 111-122.