#include <string>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <sdo/LookupTable.hpp>
#include <spline/PiecewisePolynomial.hpp>
#include <spline/SplineApproximation.hpp>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>

int main( int argc, char const *argv[] )
{
   if( argc != 2 )
   {
      std::cout << "Expected lookups number as argument" << std::endl;
      return 1;
   }

   std::ifstream lookupsDat( "lookups.dat" );
   long i = std::count( std::istreambuf_iterator<char>( lookupsDat ),
                        std::istreambuf_iterator<char>(), '\n' );
   lookupsDat.seekg( 0 );
   std::string lookupsLine;
   sdo::LookupTable lookup;
   unsigned j = boost::lexical_cast<unsigned>( argv[1] );
   i = 0;
   double minkntdistance = 1e-7;
   double delta = 2;
   double eps = 1e-7;
   double max_rel_err = 0.01;

   while( std::getline( lookupsDat, lookupsLine ) )
   {
      std::size_t pos = lookupsLine.find( '=' );
      if( pos != std::string::npos )
      {
         std::string setting = lookupsLine.substr( 0, pos );
         std::string valstring = lookupsLine.substr( pos + 1 );
         boost::trim( setting );
         boost::trim( valstring );
         double val;
         try
         {
            val = boost::lexical_cast<double>( valstring );
         }
         catch( const boost::bad_lexical_cast &e )
         {
            std::cerr << "warning: bad value '" << valstring << "' for option '" << setting << "' is ignored\n";
            std::cerr << "expected a real value\n";
            continue;
         }

         if( boost::iequals( setting, "max_mixed_err" ) )
         {
            max_rel_err = val;
         }
         else if( boost::iequals( setting, "min_knot_distance" ) )
         {
            minkntdistance = val;
         }
         else if( boost::iequals( setting, "mixed_err_delta" ) )
         {
            delta = val;
         }
         else if( boost::iequals( setting, "obj_tolerance" ) )
         {
            eps = val;
         }
         else
         {
            std::cerr << "warning: unknown option ignored '" << setting << "'\n";
            std::cerr << "supported options are: max_mixed_err, min_knot_distance, mixed_err_delta, obj_tolerance\n";
         }
         continue;
      }
      if( i == j )
      {
         double x, y;
         std::cout << std::endl << "LOOKUP " << i << ":" << std::endl;
         std::istringstream lineStream( lookupsLine );

         while( ( lineStream >> x ) && ( lineStream >> y ) )
         {
            std::cout << "\t" << "Added (" << x << "," << y << ")" << std::endl;
            lookup.addPoint( x, y );
         }

         break;
      }

      ++i;
   }

   std::ofstream data_smooth( "data_smooth.dat" );
   std::ofstream data_linear( "data_linear.dat" );
   std::ofstream data_err( "data_err.dat" );
   std::ofstream data_knots( "data_knots.dat" );
   data_linear << std::setprecision( 17 );
   data_smooth << std::setprecision( 17 );
   data_err << std::setprecision( 17 );
   data_knots << std::setprecision( 17 );
   double a, b, h;
   a = lookup.begin()->get<0>();
   b = ( lookup.end() - 1 )->get<0>();
   double xtd = ( b - a ) / 2;
   a -= xtd;
   b += xtd;

   std::cout << a << " to " << b << std::endl;


   data_linear << a << "\t" << lookup.begin()->get<1>() << std::endl;

   for( sdo::LookupTable::iterator it = lookup.begin(); it != lookup.end(); ++it )
   {
      data_linear << it->get<0>() << "\t" << it->get<1>() << std::endl;
   }

   data_linear << b << "\t" << ( lookup.end() - 1 )->get<1>() << std::endl;
   spline::BSplineCurve<1, double> linear(lookup.getXvals(), lookup.getYvals());
   double t = omp_get_wtime();
   double hmin = 1e-7;
   double rel_err = max_rel_err;
   double hmax = ( b - a ) / 4;
   auto pp = spline::ApproximatePiecewiseLinear(
                linear,
                a,
                b,
                rel_err,
                delta,
                eps,
                minkntdistance
             );

   double curveps = 1e-2;

   printf( "Fitting took %g seconds.\n", omp_get_wtime() - t );

   double m = ( b - a );

   for( std::size_t i = 0; i < pp.numIntervals(); ++i )
   {
      m = std::min( pp.getSupremum( i ) - pp.getInfimum( i ), m );
      data_knots << boost::lexical_cast<std::string>( pp.getInfimum( i ) ) << "\t"
                 << boost::lexical_cast<std::string>( pp( pp.getInfimum( i ) ) ) << std::endl;
   }

   std::cout << "Curve has " << pp.numIntervals() << " intervals. The smallest has size " << std::setprecision( 17 ) <<  m << "\n";
   std::cout << "Maximum mixed error is " << rel_err << " (delta=" << delta << ")\n";
   double p = a;
   double dcurr = pp.derivative<1>( p );
   double d2curr = pp.derivative<2>( p );
   double curr = pp( p );

   while( p <= b )
   {
      data_smooth << boost::lexical_cast<std::string>( p ) << "\t"
                  << boost::lexical_cast<std::string>( curr ) << std::endl;
      data_err << boost::lexical_cast<std::string>( p ) << "\t"
               <<  boost::lexical_cast<std::string>( std::abs( ( pp( p ) - lookup( p ) ) / ( std::abs( lookup( p ) ) + delta ) ) ) << "\n";

      h = hmax;
      double dnext;
      double d2next;
      double next;

      do
      {
         dnext = pp.derivative<1>( p + h );
         d2next = pp.derivative<2>( p + h );
         next = pp( p + h );

         if( std::abs( curr - next ) / ( curveps + curr ) <= curveps &&
               std::abs( dcurr - dnext ) / ( curveps + dcurr ) <= curveps &&
               std::abs( d2curr - d2next ) / ( curveps + d2curr ) <= curveps )
            break;

         h = std::max( h / 1.5, hmin );
      }
      while( h > hmin );

      if( b - p > eps )
         h = std::min( b - p, h );

      p += h;
      d2curr = d2next;
      curr = next;
      dcurr = dnext;
   }


}
