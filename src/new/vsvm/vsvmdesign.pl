#!/usr/bin/perl
#
# Author: Tilo Buschmann
#
# I am sorry for the shoddy code, but I did not program perl for quite some
# time and it is an obsolete language

use strict;

use Getopt::Long;
use Pod::Usage;

my $in_design       = "";
my $out_design      = "design_single.txt";
my $out_conversion  = "conversion.txt";
my $help  = 0;
my $man   = 0;

GetOptions ('help|?'            => \$help, 
            'man'               => \$man,
            'out_design=s'      => \$out_design,
            'out_conversion=s'  => \$out_conversion,
        ) or pod2usage(-verbose => 0);

pod2usage(-verbose => 1) if $help;
pod2usage(-verbose => 2) if $man;

pod2usage("$0: No files given.")  if ((@ARGV == 0) && (-t STDIN));

$in_design = $ARGV[0];

# 1) Design mit Nummer pro Kondition zu einem Design mit Nummer pro Event


open INPUT_DESIGN,$in_design                or die $!;
open OUTPUT_DESIGN,">",$out_design          or die $!;
open OUTPUT_CONVERSION,">",$out_conversion  or die $!;

my $line_number = 1;

while (<INPUT_DESIGN>) {
  if (/^\s*\%/ || /^\s*$/) {
    # Comment or empty
    print OUTPUT_DESIGN;
  } else {
    # Entry
    chomp;
    split /\t/;  
    print OUTPUT_DESIGN "$line_number\t$_[1]\t$_[2]\t$_[3]\n";
    print OUTPUT_CONVERSION "$line_number\t$_[0]\n";
    $line_number++;
  }
}

__END__

=head1 NAME

vsvmdesign.pl - Convert a design file to a single trial design file and another file that contains the original information on the connection between trial and class

=head1 SYNOPSIS

vsvmdesign.pl [options] [design_file]

 Options:
   -help            brief help message
   -man             full documentation
   -out_design      ouput design file
   -out_conversion  output conversion file

=head1 OPTIONS

=over 8

=item B<-help>

Print a brief help message and exits.

=item B<-man>

Prints the manual page and exits.

=item B<-out_design>

Name of the first level design file that needs to be converted

=item B<-out_conversion>

Name of the conversion file between single trial numbers and classes

=back

=head1 DESCRIPTION

  vsvmdesign.pl design.txt
  vgendesign -in design_single.txt -out design_single.v -tr 1 -ntimesteps 1200 -deriv 0
  vcolorglm -in wrdata.v -design design_single.v -out beta_single.v
  vsplitbeta -in beta_single.v -base "separate_betas" -conversion conversion.txt

=cut
