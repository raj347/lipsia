/****************************************************************
 *
 * Copyright (C) 2010 Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Author: Erik Tuerke, tuerke@cbs.mpg.de, 2010
 *
 *****************************************************************/


#include "isisRegistrationFactory3D.hpp"

namespace isis
{
namespace registration
{

template<class TFixedImageType, class TMovingImageType>
RegistrationFactory3D<TFixedImageType, TMovingImageType>::RegistrationFactory3D()
{
	m_RegistrationObject = RegistrationMethodType::New();
	this->Reset();
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::Reset(
	void )
{
	//boolean settings
	optimizer.REGULARSTEPGRADIENTDESCENT = false;
	optimizer.VERSORRIGID3D = false;
	optimizer.LBFGSBOPTIMIZER = false;
	optimizer.AMOEBA = false;
	optimizer.POWELL = false;
	transform.TRANSLATION = false;
	transform.VERSORRIGID = false;
	transform.AFFINE = false;
	transform.CENTEREDAFFINE = false;
	transform.BSPLINEDEFORMABLETRANSFORM = false;
	transform.RIGID3D = false;
	metric.MATTESMUTUALINFORMATION = false;
	metric.NORMALIZEDCORRELATION = false;
	metric.VIOLAWELLSMUTUALINFORMATION = false;
	metric.MEANSQUARE = false;
	metric.MUTUALINFORMATIONHISTOGRAM = false;
	interpolator.BSPLINE = false;
	interpolator.LINEAR = false;
	interpolator.NEARESTNEIGHBOR = false;
	m_InitialTransformIsSet = false;
	UserOptions.PRINTRESULTS = false;
	UserOptions.NumberOfIterations = 1000;
	UserOptions.NumberOfBins = 100;
	UserOptions.PixelDensity = 0.01;
	UserOptions.USEOTSUTHRESHOLDING = false;
	UserOptions.BSplineGridSize = 5;
	UserOptions.BSplineBound = 100;
	UserOptions.INITIALIZECENTEROFF = false;
	UserOptions.INITIALIZEMASSOFF = false;
	UserOptions.PREALIGN = false;
	UserOptions.PREALIGNPRECISION = 5;
	UserOptions.NumberOfThreads = 1;
	UserOptions.MattesMutualInitializeSeed = 1;
	UserOptions.SHOWITERATIONATSTEP = 1;
	UserOptions.USEMASK = false;
	UserOptions.LANDMARKINITIALIZE = false;
	UserOptions.CoarseFactor = 1;
	UserOptions.ROTATIONSCALE = -1; // not set
	UserOptions.TRANSLATIONSCALE = -1; //not set
	m_NumberOfParameters = 0;
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetFixedImage(
	FixedImagePointer fixedImage )
{
	m_FixedImage = fixedImage;
	m_RegistrationObject->SetFixedImage( m_FixedImage );
	m_FixedImageRegion = m_FixedImage->GetLargestPossibleRegion();
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetMovingImage(
	MovingImagePointer movingImage )
{
	m_MovingImage = movingImage;
	m_RegistrationObject->SetMovingImage( m_MovingImage );
	m_MovingImageRegion = m_MovingImage->GetLargestPossibleRegion();
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetMetric(
	eMetricType e_metric )
{
	switch ( e_metric ) {
	case MattesMutualInformationMetric:
		metric.MATTESMUTUALINFORMATION = true;
		m_MattesMutualInformationMetric = MattesMutualInformationMetricType::New();
		m_RegistrationObject->SetMetric( m_MattesMutualInformationMetric );
		break;
	case ViolaWellsMutualInformationMetric:
		metric.VIOLAWELLSMUTUALINFORMATION = true;
		m_ViolaWellsMutualInformationMetric = ViolaWellsMutualInformationMetricType::New();
		m_RegistrationObject->SetMetric( m_ViolaWellsMutualInformationMetric );
		break;
	case MutualInformationHistogramMetric:
		metric.MUTUALINFORMATIONHISTOGRAM = true;
		m_MutualInformationHistogramMetric = MutualInformationHistogramMetricType::New();
		m_RegistrationObject->SetMetric( m_MutualInformationHistogramMetric );
		break;
	case NormalizedCorrelationMetric:
		metric.NORMALIZEDCORRELATION = true;
		m_NormalizedCorrelationMetric = NormalizedCorrelationMetricType::New();
		m_RegistrationObject->SetMetric( m_NormalizedCorrelationMetric );
		break;
	case MeanSquareMetric:
		metric.MEANSQUARE = true;
		m_MeanSquareMetric = MeanSquareImageToImageMetricType::New();
		m_RegistrationObject->SetMetric( m_MeanSquareMetric );
		break;
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetInterpolator(
	eInterpolationType e_interpolator )
{
	switch ( e_interpolator ) {
	case LinearInterpolator:
		interpolator.LINEAR = true;
		m_LinearInterpolator = LinearInterpolatorType::New();
		m_RegistrationObject->SetInterpolator( m_LinearInterpolator );
		break;
	case BSplineInterpolator:
		interpolator.BSPLINE = true;
		m_BSplineInterpolator = BSplineInterpolatorType::New();
		m_RegistrationObject->SetInterpolator( m_BSplineInterpolator );
		break;
	case NearestNeighborInterpolator:
		interpolator.NEARESTNEIGHBOR = true;
		m_NearestNeighborInterpolator = NearestNeighborInterpolatorType::New();
		m_RegistrationObject->SetInterpolator( m_NearestNeighborInterpolator );
		break;
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetTransform(
	eTransformType e_transform )
{
	switch ( e_transform ) {
	case TranslationTransform:
		transform.TRANSLATION = true;
		m_TranslationTransform = TranslationTransformType::New();
		m_RegistrationObject->SetTransform( m_TranslationTransform );
		break;
	case VersorRigid3DTransform:
		transform.VERSORRIGID = true;
		m_VersorRigid3DTransform = VersorRigid3DTransformType::New();
		m_RegistrationObject->SetTransform( m_VersorRigid3DTransform );
		break;
	case AffineTransform:
		transform.AFFINE = true;
		m_AffineTransform = AffineTransformType::New();
		m_RegistrationObject->SetTransform( m_AffineTransform );
		break;
	case CenteredAffineTransform:
		transform.CENTEREDAFFINE = true;
		m_CenteredAffineTransform = CenteredAffineTransformType::New();
		m_RegistrationObject->SetTransform( m_CenteredAffineTransform );
		break;
	case BSplineDeformableTransform:
		transform.BSPLINEDEFORMABLETRANSFORM = true;
		m_BSplineTransform = BSplineTransformType::New();
		m_RegistrationObject->SetTransform( m_BSplineTransform );
		break;
	case Rigid3DTransform:
		transform.RIGID3D = true;
		m_Rigid3DTransform = Rigid3DTransformType::New();
		m_RegistrationObject->SetTransform( m_Rigid3DTransform );
		break;
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetOptimizer(
	eOptimizerType e_optimizer )
{
	switch ( e_optimizer ) {
	case RegularStepGradientDescentOptimizer:
		optimizer.REGULARSTEPGRADIENTDESCENT = true;
		m_RegularStepGradientDescentOptimizer = RegularStepGradientDescentOptimizerType::New();
		m_RegistrationObject->SetOptimizer( m_RegularStepGradientDescentOptimizer );
		break;
	case VersorRigidOptimizer:
		optimizer.VERSORRIGID3D = true;
		m_VersorRigid3DTransformOptimizer = VersorRigid3DTransformOptimizerType::New();
		m_RegistrationObject->SetOptimizer( m_VersorRigid3DTransformOptimizer );
		break;
	case LBFGSBOptimizer:
		optimizer.LBFGSBOPTIMIZER = true;
		m_LBFGSBOptimizer = LBFGSBOptimizerType::New();
		m_RegistrationObject->SetOptimizer( m_LBFGSBOptimizer );
		break;
	case AmoebaOptimizer:
		optimizer.AMOEBA = true;
		m_AmoebaOptimizer = AmoebaOptimizerType::New();
		m_RegistrationObject->SetOptimizer( m_AmoebaOptimizer );
		break;
	case PowellOptimizer:
		optimizer.POWELL = true;
		m_PowellOptimizer = PowellOptimizerType::New();
		m_RegistrationObject->SetOptimizer( m_PowellOptimizer );
		break;
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++parameter setting methods++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::UpdateParameters()
{

	//transform parameters:
	this->SetUpTransform();
	//optimizer parameters:
	this->SetUpOptimizer();
	//metric parameters;
	this->SetUpMetric();

}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetUpOptimizer()
{
	if ( optimizer.REGULARSTEPGRADIENTDESCENT ) {
		//setting up the regular step gradient descent optimizer...
		RegularStepGradientDescentOptimizerType::ScalesType optimizerScaleRegularStepGradient( m_NumberOfParameters );

		if ( transform.VERSORRIGID or transform.CENTEREDAFFINE or transform.AFFINE or transform.BSPLINEDEFORMABLETRANSFORM  or transform.RIGID3D ) {
			//...for the rigid transform
			//number of parameters are dependent on the dimension of the images (2D: 4 parameter, 3D: 6 parameters)
			if ( transform.VERSORRIGID ) {
				//rotation
				if ( UserOptions.ROTATIONSCALE == -1 ) {
					UserOptions.ROTATIONSCALE = 1.0 / 1.0;

				}

				optimizerScaleRegularStepGradient[0] = UserOptions.ROTATIONSCALE;
				optimizerScaleRegularStepGradient[1] = UserOptions.ROTATIONSCALE;
				optimizerScaleRegularStepGradient[2] = UserOptions.ROTATIONSCALE;

				//translation
				if ( UserOptions.TRANSLATIONSCALE == -1 ) {
					typename FixedImageType::SizeType imageSize = m_FixedImageRegion.GetSize();
					UserOptions.TRANSLATIONSCALE =
						( sqrt( imageSize[0] * imageSize[0] + imageSize[1] * imageSize[1] + imageSize[2] * imageSize[2] ) );
				}

				optimizerScaleRegularStepGradient[3] = 1.0 / UserOptions.TRANSLATIONSCALE;
				optimizerScaleRegularStepGradient[4] = 1.0 / UserOptions.TRANSLATIONSCALE;
				optimizerScaleRegularStepGradient[5] = 1.0 / UserOptions.TRANSLATIONSCALE;
			}

			if ( transform.RIGID3D ) {
				for ( unsigned short i = 0; i < 9; i++ ) {
					optimizerScaleRegularStepGradient[i] = 1.0;
				}

				optimizerScaleRegularStepGradient[9] = 1.0;
				optimizerScaleRegularStepGradient[10] = 1.0;
				optimizerScaleRegularStepGradient[11] = 1.0;
			}

			if ( transform.BSPLINEDEFORMABLETRANSFORM or transform.AFFINE or transform.CENTEREDAFFINE or transform.TRANSLATION ) {
				optimizerScaleRegularStepGradient.Fill( 1.0 );
			}

			if ( transform.AFFINE ) {
				optimizerScaleRegularStepGradient[9] = 1.0 / UserOptions.TRANSLATIONSCALE;
				optimizerScaleRegularStepGradient[10] = 1.0 / UserOptions.TRANSLATIONSCALE;
				optimizerScaleRegularStepGradient[11] = 1.0 / UserOptions.TRANSLATIONSCALE;
			}

			m_RegularStepGradientDescentOptimizer->SetMaximumStepLength( 0.1 * UserOptions.CoarseFactor );
			m_RegularStepGradientDescentOptimizer->SetMinimumStepLength( 0.0001 * UserOptions.CoarseFactor );
			m_RegularStepGradientDescentOptimizer->SetScales( optimizerScaleRegularStepGradient );
			m_RegularStepGradientDescentOptimizer->SetNumberOfIterations( UserOptions.NumberOfIterations );
			m_RegularStepGradientDescentOptimizer->SetRelaxationFactor( 0.9 );
			m_RegularStepGradientDescentOptimizer->SetGradientMagnitudeTolerance( 0.00001 );
			m_RegularStepGradientDescentOptimizer->SetMinimize( true );

			if ( transform.BSPLINEDEFORMABLETRANSFORM ) {
				m_RegularStepGradientDescentOptimizer->SetMaximumStepLength( 1.0 );
			}
		}

		if ( metric.MEANSQUARE or metric.MATTESMUTUALINFORMATION or metric.VIOLAWELLSMUTUALINFORMATION or metric.MUTUALINFORMATIONHISTOGRAM ) {
			m_RegularStepGradientDescentOptimizer->MinimizeOn();
		}
	}

	if ( optimizer.VERSORRIGID3D ) {
		VersorRigid3DTransformOptimizerType::ScalesType optimizerScaleVersorRigid3D( m_NumberOfParameters );

		if ( transform.VERSORRIGID ) {
			optimizerScaleVersorRigid3D[0] = 1.0;
			optimizerScaleVersorRigid3D[1] = 1.0;
			optimizerScaleVersorRigid3D[2] = 1.0;
			optimizerScaleVersorRigid3D[3] = 1.0 / 1000.0;
			optimizerScaleVersorRigid3D[4] = 1.0 / 1000.0;
			optimizerScaleVersorRigid3D[5] = 1.0 / 1000.0;
		}

		m_VersorRigid3DTransformOptimizer->SetMaximumStepLength( 0.1 * UserOptions.CoarseFactor );
		m_VersorRigid3DTransformOptimizer->SetMinimumStepLength( 0.0001 *  UserOptions.CoarseFactor );
		m_VersorRigid3DTransformOptimizer->SetScales( optimizerScaleVersorRigid3D );
		m_VersorRigid3DTransformOptimizer->SetNumberOfIterations( UserOptions.NumberOfIterations );
		m_VersorRigid3DTransformOptimizer->SetRelaxationFactor( 0.9 );

		if ( metric.MEANSQUARE or metric.MATTESMUTUALINFORMATION or metric.VIOLAWELLSMUTUALINFORMATION or metric.MUTUALINFORMATIONHISTOGRAM ) {
			m_VersorRigid3DTransformOptimizer->MinimizeOn();
		}
	}

	if ( optimizer.LBFGSBOPTIMIZER ) {
		LBFGSBOptimizerType::BoundSelectionType boundSelect( m_NumberOfParameters );
		LBFGSBOptimizerType::BoundValueType lowerBound( m_NumberOfParameters );
		LBFGSBOptimizerType::BoundValueType upperBound( m_NumberOfParameters );
		boundSelect.Fill( 2 );
		lowerBound.Fill( -UserOptions.BSplineBound );
		upperBound.Fill( UserOptions.BSplineBound );
		m_LBFGSBOptimizer->SetBoundSelection( boundSelect );
		m_LBFGSBOptimizer->SetLowerBound( lowerBound );
		m_LBFGSBOptimizer->SetUpperBound( upperBound );
		m_LBFGSBOptimizer->SetCostFunctionConvergenceFactor( 1.0 );
		m_LBFGSBOptimizer->SetProjectedGradientTolerance( 1e-12 );
		m_LBFGSBOptimizer->SetMaximumNumberOfIterations( UserOptions.NumberOfIterations );
		m_LBFGSBOptimizer->SetMaximumNumberOfEvaluations( 5000 );
		m_LBFGSBOptimizer->SetMaximumNumberOfCorrections( 240 );

		if ( metric.MEANSQUARE or metric.MATTESMUTUALINFORMATION or metric.VIOLAWELLSMUTUALINFORMATION or metric.MUTUALINFORMATIONHISTOGRAM ) {
			m_LBFGSBOptimizer->MinimizeOn();
			m_LBFGSBOptimizer->MaximizeOff();
		}
	}

	if ( optimizer.AMOEBA ) {
		AmoebaOptimizerType::ParametersType simplexDelta( m_NumberOfParameters );
		//simplexDelta.Fill(5.0);
		m_AmoebaOptimizer->AutomaticInitialSimplexOn();
		m_AmoebaOptimizer->SetInitialSimplexDelta( simplexDelta );
		m_AmoebaOptimizer->SetMaximumNumberOfIterations( UserOptions.NumberOfIterations );
		m_AmoebaOptimizer->SetParametersConvergenceTolerance( 1e-10 );
		m_AmoebaOptimizer->SetFunctionConvergenceTolerance( 1e-10 );

		if ( metric.MEANSQUARE or metric.MATTESMUTUALINFORMATION or metric.VIOLAWELLSMUTUALINFORMATION or metric.MUTUALINFORMATIONHISTOGRAM ) {
			m_AmoebaOptimizer->MinimizeOn();
		}

		if ( metric.NORMALIZEDCORRELATION ) {
			m_AmoebaOptimizer->MaximizeOn();
		}
	}

	if ( optimizer.POWELL ) {
	};
}




template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D <
TFixedImageType, TMovingImageType >::prealign()
{
	m_MattesMutualInformationMetric->SetNumberOfThreads( UserOptions.NumberOfThreads );
	m_VersorRigid3DTransform = VersorRigid3DTransformType::New();
	m_RigidInitializer = RigidCenteredTransformInitializerType::New();
	m_RigidInitializer->SetTransform( m_VersorRigid3DTransform );
	m_RigidInitializer->SetFixedImage( m_FixedImage );
	m_RigidInitializer->SetMovingImage( m_MovingImage );
	m_RigidInitializer->GeometryOn();
	m_RigidInitializer->InitializeTransform();

	if( !metric.MATTESMUTUALINFORMATION ) {
		m_MattesMutualInformationMetric = MattesMutualInformationMetricType::New();
	}

	m_MattesMutualInformationMetric->SetMovingImage( m_MovingImage );
	m_MattesMutualInformationMetric->SetFixedImage( m_FixedImage );
	m_MattesMutualInformationMetric->SetFixedImageRegion( m_FixedImageRegion );
	m_MattesMutualInformationMetric->SetTransform( m_VersorRigid3DTransform );
	m_MattesMutualInformationMetric->SetNumberOfSpatialSamples( m_FixedImageRegion.GetNumberOfPixels()
			* UserOptions.PixelDensity / 2 );
	m_MattesMutualInformationMetric->SetNumberOfHistogramBins( UserOptions.NumberOfBins / 2 );
	m_MattesMutualInformationMetric->SetInterpolator( m_LinearInterpolator );
	typename VersorRigid3DTransformType::ParametersType params = m_VersorRigid3DTransform->GetParameters();
	typename VersorRigid3DTransformType::ParametersType searchParams = m_VersorRigid3DTransform->GetParameters();
	typename VersorRigid3DTransformType::ParametersType newParams = m_VersorRigid3DTransform->GetParameters();
	m_MattesMutualInformationMetric->Initialize();
	typename MovingImageType::SizeType movingImageSize = m_MovingImageRegion.GetSize();
	typename MovingImageType::SpacingType movingImageSpacing = m_MovingImage->GetSpacing();
	short prec = 5;
	float ratio = 0.3;
	float minMaxX = ratio * movingImageSize[0] * movingImageSpacing[0];
	float minMaxY = ratio * movingImageSize[1] * movingImageSpacing[1];
	float minMaxZ = ratio * movingImageSize[2] * movingImageSpacing[2];
	float stepSizeX = minMaxX / ( float )prec;
	float stepSizeY = minMaxY / ( float )prec;
	float stepSizeZ = minMaxZ / ( float )prec;
	double value = 0;
	double metricValue = 0;

	for ( float x = -minMaxX; x <= minMaxX; x += stepSizeX ) {
		for ( float y = -minMaxY; y <= minMaxY; y += stepSizeY ) {
			for ( float z = -minMaxZ; z <= minMaxZ; z += stepSizeZ ) {
				searchParams[3] = params[3] +  x;
				searchParams[4] = params[4] +  y;
				searchParams[5] = params[5] +  z;
				metricValue = static_cast<double>( m_MattesMutualInformationMetric->GetValue(  searchParams ) );

				if ( value >  metricValue ) {
					value = metricValue;
					newParams = searchParams;
				}
			}
		}
	}

	m_VersorRigid3DTransform->SetParameters( newParams );
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetUpTransform()
{
	if ( UserOptions.PREALIGN ) {
		std::cout << "Prealigning..." << std::endl;
		prealign();
	}

	//initialize transform
	if ( !UserOptions.INITIALIZEMASSOFF or !UserOptions.INITIALIZECENTEROFF ) {
		if ( transform.TRANSLATION ) {
			m_VersorRigid3DTransform = VersorRigid3DTransformType::New();
			m_RigidInitializer = RigidCenteredTransformInitializerType::New();
			m_RigidInitializer->SetTransform( m_VersorRigid3DTransform );
			m_RigidInitializer->SetFixedImage( m_FixedImage );
			m_RigidInitializer->SetMovingImage( m_MovingImage );

			if ( !UserOptions.INITIALIZECENTEROFF )
				m_RigidInitializer->GeometryOn();

			if ( !UserOptions.INITIALIZEMASSOFF )
				m_RigidInitializer->MomentsOn();

			m_RigidInitializer->InitializeTransform();
			VersorRigid3DTransformType::ParametersType parameters( FixedImageDimension );
			parameters[0] = m_VersorRigid3DTransform->GetTranslation()[0];
			parameters[1] = m_VersorRigid3DTransform->GetTranslation()[1];
			parameters[2] = m_VersorRigid3DTransform->GetTranslation()[2];
			m_TranslationTransform->SetParameters( parameters );
		}

		if ( transform.VERSORRIGID ) {
			m_RigidInitializer = RigidCenteredTransformInitializerType::New();
			m_RigidInitializer->SetTransform( m_VersorRigid3DTransform );
			m_RigidInitializer->SetFixedImage( m_FixedImage );
			m_RigidInitializer->SetMovingImage( m_MovingImage );

			if ( !UserOptions.INITIALIZECENTEROFF )
				m_RigidInitializer->GeometryOn();

			if ( !UserOptions.INITIALIZEMASSOFF )
				m_RigidInitializer->MomentsOn();

			m_RigidInitializer->InitializeTransform();
		}

		if ( transform.AFFINE ) {
			m_AffineInitializer = AffineCenteredTransformInitializerType::New();
			m_AffineInitializer->SetTransform( m_AffineTransform );
			m_AffineInitializer->SetFixedImage( m_FixedImage );
			m_AffineInitializer->SetMovingImage( m_MovingImage );
			m_AffineInitializer->GeometryOn();

			if ( !UserOptions.INITIALIZECENTEROFF )
				m_AffineInitializer->GeometryOn();

			if ( !UserOptions.INITIALIZEMASSOFF )
				m_AffineInitializer->MomentsOn();

			m_AffineInitializer->InitializeTransform();
		}
	}

	if ( transform.BSPLINEDEFORMABLETRANSFORM ) {
		typedef typename BSplineTransformType::RegionType BSplineRegionType;
		typedef typename BSplineTransformType::SpacingType BSplineSpacingType;
		typedef typename BSplineTransformType::OriginType BSplineOriginType;
		typedef typename BSplineTransformType::DirectionType BSplineDirectionType;
		BSplineRegionType bsplineRegion;
		typename BSplineRegionType::SizeType gridSizeOnImage;
		typename BSplineRegionType::SizeType gridBorderSize;
		typename BSplineRegionType::SizeType totalGridSize;
		gridSizeOnImage.Fill( UserOptions.BSplineGridSize );
		gridBorderSize.Fill( 3 ); //Border for spline order = 3 (1 lower, 2 upper)
		totalGridSize = gridSizeOnImage + gridBorderSize;
		bsplineRegion.SetSize( totalGridSize );
		BSplineSpacingType bsplineSpacing = m_FixedImage->GetSpacing();
		typename FixedImageType::SizeType fixedImageSize = m_FixedImage->GetBufferedRegion().GetSize();

		for ( unsigned int r = 0; r < FixedImageDimension; r++ ) {
			bsplineSpacing[r] *= static_cast<double> ( fixedImageSize[r] - 1 ) / static_cast<double> ( gridSizeOnImage[r]
								 - 1 );
		}

		BSplineDirectionType bsplineDirection = m_FixedImage->GetDirection();
		BSplineSpacingType gridOriginOffset = bsplineDirection * bsplineSpacing;
		BSplineOriginType bsplineOrigin = m_FixedImage->GetOrigin();
		bsplineOrigin = bsplineOrigin - gridOriginOffset;
		m_BSplineTransform->SetGridSpacing( bsplineSpacing );
		m_BSplineTransform->SetGridOrigin( bsplineOrigin );
		m_BSplineTransform->SetGridRegion( bsplineRegion );
		m_BSplineTransform->SetGridDirection( bsplineDirection );
		m_NumberOfParameters = m_BSplineTransform->GetNumberOfParameters();
		m_BSplineParameters.SetSize( m_NumberOfParameters );
		m_BSplineParameters.Fill( 0.0 );
		m_BSplineTransform->SetParameters( m_BSplineParameters );
		m_RegistrationObject->SetInitialTransformParameters( m_BSplineTransform->GetParameters() );
	}

	if ( transform.AFFINE ) {
		m_NumberOfParameters = m_AffineTransform->GetNumberOfParameters();
		m_RegistrationObject->SetInitialTransformParameters( m_AffineTransform->GetParameters() );
	}

	if ( transform.CENTEREDAFFINE ) {
		m_NumberOfParameters = m_CenteredAffineTransform->GetNumberOfParameters();
		m_RegistrationObject->SetInitialTransformParameters( m_CenteredAffineTransform->GetParameters() );
	}

	if ( transform.VERSORRIGID ) {
		m_NumberOfParameters = m_VersorRigid3DTransform->GetNumberOfParameters();
		m_RegistrationObject->SetInitialTransformParameters( m_VersorRigid3DTransform->GetParameters() );
	}

	if ( transform.RIGID3D ) {
		m_NumberOfParameters = m_Rigid3DTransform->GetNumberOfParameters();
		m_RegistrationObject->SetInitialTransformParameters( m_Rigid3DTransform->GetParameters() );
	}

	if ( transform.TRANSLATION ) {
		m_NumberOfParameters = m_TranslationTransform->GetNumberOfParameters();
		m_RegistrationObject->SetInitialTransformParameters( m_TranslationTransform->GetParameters() );
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetUpMetric()
{
	if ( metric.MATTESMUTUALINFORMATION ) {
		//setting up the mattes mutual information metric
		m_MattesMutualInformationMetric->SetFixedImage( m_FixedImage );
		m_MattesMutualInformationMetric->SetMovingImage( m_MovingImage );
		m_MattesMutualInformationMetric->SetFixedImageRegion( m_FixedImageRegion );
		m_MattesMutualInformationMetric->SetNumberOfSpatialSamples( m_FixedImageRegion.GetNumberOfPixels()
				* UserOptions.PixelDensity );
		m_MattesMutualInformationMetric->SetNumberOfHistogramBins( UserOptions.NumberOfBins );
		m_MattesMutualInformationMetric->ReinitializeSeed( UserOptions.MattesMutualInitializeSeed );

		if ( transform.BSPLINEDEFORMABLETRANSFORM ) {
			m_MattesMutualInformationMetric->SetUseCachingOfBSplineWeights( true );
		}
	}

	if ( metric.VIOLAWELLSMUTUALINFORMATION ) {
		//set up the filters
		m_FixedGaussianFilter = DiscreteGaussianImageFitlerType::New();
		m_MovingGaussianFilter = DiscreteGaussianImageFitlerType::New();
		m_FixedNormalizeImageFilter = FixedNormalizeImageFilterType::New();
		m_MovingNormalizeImageFilter = MovingNormalizeImageFilterType::New();
		m_FixedGaussianFilter->SetVariance( 2.0 );
		m_MovingGaussianFilter->SetVariance( 2.0 );
		//pipelining the images: NormalizeImageFilter -> GaussianImageFilter -> RegistrationMethod
		m_FixedNormalizeImageFilter->SetInput( m_FixedImage );
		m_MovingNormalizeImageFilter->SetInput( m_MovingImage );
		m_FixedGaussianFilter->SetInput( m_FixedNormalizeImageFilter->GetOutput() );
		m_MovingGaussianFilter->SetInput( m_MovingNormalizeImageFilter->GetOutput() );
		m_ViolaWellsMutualInformationMetric->SetFixedImage( m_FixedGaussianFilter->GetOutput() );
		m_ViolaWellsMutualInformationMetric->SetMovingImage( m_MovingGaussianFilter->GetOutput() );
		m_ViolaWellsMutualInformationMetric->SetFixedImageRegion( m_FixedImageRegion );
		m_ViolaWellsMutualInformationMetric->SetNumberOfSpatialSamples( m_FixedImageRegion.GetNumberOfPixels()
				* UserOptions.PixelDensity );
		m_ViolaWellsMutualInformationMetric->SetFixedImageStandardDeviation( 0.4 );
		m_ViolaWellsMutualInformationMetric->SetMovingImageStandardDeviation( 0.4 );
	}

	if ( metric.MUTUALINFORMATIONHISTOGRAM ) {
		typename MutualInformationHistogramMetricType::HistogramSizeType histogramSize;
		histogramSize[0] = UserOptions.NumberOfBins;
		histogramSize[1] = UserOptions.NumberOfBins;
		m_MutualInformationHistogramMetric->SetHistogramSize( histogramSize );

		if ( optimizer.AMOEBA ) {
			m_MutualInformationHistogramMetric->ComputeGradientOff();
		}
	}

	if ( metric.NORMALIZEDCORRELATION ) {
		//setting up the normalized correlation metric
		m_NormalizedCorrelationMetric->SetFixedImage( m_FixedImage );
		m_NormalizedCorrelationMetric->SetMovingImage( m_MovingImage );
		m_NormalizedCorrelationMetric->SetFixedImageRegion( m_FixedImageRegion );
	}

	if ( metric.MEANSQUARE ) {
		m_MeanSquareMetric->SetFixedImage( m_FixedImage );
		m_MeanSquareMetric->SetMovingImage( m_MovingImage );
		m_MeanSquareMetric->SetFixedImageRegion( m_FixedImageRegion );
	}
}

template<class TFixedImageType, class TMovingImageType>
typename RegistrationFactory3D<TFixedImageType, TMovingImageType>::OutputImagePointer RegistrationFactory3D <
TFixedImageType, TMovingImageType >::GetRegisteredImage(
	void )
{
	m_ResampleFilter = ResampleFilterType::New();
	m_ImageCaster = ImageCasterType::New();
	typename RegistrationMethodType::OptimizerType::ParametersType finalParameters =
		m_RegistrationObject->GetLastTransformParameters();
	m_RegistrationObject->GetTransform()->SetParameters( finalParameters );
	m_ResampleFilter->SetInput( m_MovingImage );
	m_ResampleFilter->SetTransform( m_RegistrationObject->GetTransform() );
	m_ResampleFilter->SetOutputOrigin( m_FixedImage->GetOrigin() );
	m_ResampleFilter->SetSize( m_FixedImage->GetLargestPossibleRegion().GetSize() );
	m_ResampleFilter->SetOutputSpacing( m_FixedImage->GetSpacing() );
	m_ResampleFilter->SetOutputDirection( m_FixedImage->GetDirection() );
	m_ResampleFilter->SetDefaultPixelValue( 0 );
	m_ImageCaster->SetInput( m_ResampleFilter->GetOutput() );
	m_OutputImage = m_ImageCaster->GetOutput();
	m_ImageCaster->Update();
	return m_OutputImage;
}

template<class TFixedImageType, class TMovingImageType>
typename RegistrationFactory3D<TFixedImageType, TMovingImageType>::ConstTransformBasePointer RegistrationFactory3D <
TFixedImageType, TMovingImageType >::GetTransform(
	void )
{
	return m_RegistrationObject->GetOutput()->Get();
}

template<class TFixedImageType, class TMovingImageType>
typename RegistrationFactory3D<TFixedImageType, TMovingImageType>::DeformationFieldPointer RegistrationFactory3D <
TFixedImageType, TMovingImageType >::GetTransformVectorField(
	void )
{
	m_DeformationField = DeformationFieldType::New();
	m_DeformationField->SetRegions( m_FixedImageRegion );
	m_DeformationField->SetOrigin( m_FixedImage->GetOrigin() );
	m_DeformationField->SetSpacing( m_FixedImage->GetSpacing() );
	m_DeformationField->SetDirection( m_FixedImage->GetDirection() );
	m_DeformationField->Allocate();
	typedef itk::ImageRegionIterator<DeformationFieldType> DeformationFieldIteratorType;
	DeformationFieldIteratorType fi( m_DeformationField, m_FixedImageRegion );
	fi.GoToBegin();
	typename itk::Transform<double, FixedImageDimension, MovingImageDimension>::InputPointType fixedPoint;
	typename itk::Transform<double, FixedImageDimension, MovingImageDimension>::OutputPointType movingPoint;
	typename DeformationFieldType::IndexType index;
	VectorType displacement;

	while ( !fi.IsAtEnd() ) {
		index = fi.GetIndex();
		m_DeformationField->TransformIndexToPhysicalPoint( index, fixedPoint );
		movingPoint = m_RegistrationObject->GetOutput()->Get()->TransformPoint( fixedPoint );
		displacement = movingPoint - fixedPoint;
		fi.Set( displacement );
		++fi;
	}

	return m_DeformationField;
}

template<class TFixedImageType, class TMovingImageType>
typename RegistrationFactory3D<TFixedImageType, TMovingImageType>::RegistrationMethodPointer RegistrationFactory3D <
TFixedImageType, TMovingImageType >::GetRegistrationObject(
	void )
{
	this->UpdateParameters();
	return m_RegistrationObject;
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetInitialTransform(
	TransformBasePointer initialTransform )
{
	const char *initialTransformName = initialTransform->GetNameOfClass();

	if ( !strcmp( initialTransformName, "AffineTransform" ) and transform.BSPLINEDEFORMABLETRANSFORM ) {
		m_BSplineTransform->SetBulkTransform( dynamic_cast<AffineTransformType *>( initialTransform ) );
		m_RegistrationObject->SetInitialTransformParameters( m_BSplineTransform->GetParameters() );
	}

	if ( !strcmp( initialTransformName, "VersorRigid3DTransform" ) and transform.BSPLINEDEFORMABLETRANSFORM ) {
		m_BSplineTransform->SetBulkTransform( dynamic_cast<VersorRigid3DTransformType *> ( initialTransform ) );
	}

	if ( !strcmp( initialTransformName, "CenteredAffineTransform" ) and transform.BSPLINEDEFORMABLETRANSFORM ) {
		m_BSplineTransform->SetBulkTransform( dynamic_cast<CenteredAffineTransformType *> ( initialTransform ) );
	}

	if ( !strcmp( initialTransformName, "VersorRigid3DTransform" ) and transform.CENTEREDAFFINE ) {
		m_CenteredAffineTransform->SetTranslation(
			( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetTranslation() ) );
		m_CenteredAffineTransform->SetMatrix( ( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetMatrix() ) );
		m_RegistrationObject->SetInitialTransformParameters( m_CenteredAffineTransform->GetParameters() );
	}

	if ( !strcmp( initialTransformName, "VersorRigid3DTransform" ) and transform.AFFINE ) {
		m_AffineTransform->SetTranslation(
			( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetTranslation() ) );
		m_AffineTransform->SetMatrix( ( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetMatrix() ) );
		m_RegistrationObject->SetInitialTransformParameters( m_AffineTransform->GetParameters() );
	}

	if ( !strcmp( initialTransformName, "VersorRigid3DTransform" ) and transform.VERSORRIGID ) {
		m_VersorRigid3DTransform->SetTranslation(
			( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetTranslation() ) );
		m_VersorRigid3DTransform->SetMatrix( ( static_cast<VersorRigid3DTransformType *> ( initialTransform )->GetMatrix() ) );
		m_RegistrationObject->SetInitialTransformParameters( m_VersorRigid3DTransform->GetParameters() );
	}
}


template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetMovingPointContainer( typename RigidLandmarkBasedTransformInitializerType::LandmarkPointContainer pointContainer )
{
	m_FixedPointContainer = pointContainer;
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetFixedPointContainer( typename RigidLandmarkBasedTransformInitializerType::LandmarkPointContainer pointContainer )
{
	m_MovingPointContainer = pointContainer;
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetFixedImageMask( typename MaskObjectType::Pointer maskObject )
{
	UserOptions.USEMASK = true;
	m_MovingImageMaskObject = maskObject;
	this->SetFixedImageMask();
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::SetFixedImageMask(
	void )
{
	if ( metric.MATTESMUTUALINFORMATION ) {
		m_MattesMutualInformationMetric->SetFixedImageMask( m_MovingImageMaskObject );
		m_MattesMutualInformationMetric->SetMovingImageMask( m_MovingImageMaskObject );
	}

	if ( metric.VIOLAWELLSMUTUALINFORMATION ) {
		m_ViolaWellsMutualInformationMetric->SetFixedImageMask( m_MovingImageMaskObject );
	}

	if ( metric.MUTUALINFORMATIONHISTOGRAM ) {
		m_MutualInformationHistogramMetric->SetFixedImageMask( m_MovingImageMaskObject );
	}

	if ( metric.NORMALIZEDCORRELATION ) {
		m_NormalizedCorrelationMetric->SetFixedImageMask( m_MovingImageMaskObject );
		m_NormalizedCorrelationMetric->SetMovingImageMask( m_MovingImageMaskObject );
	}

	if ( metric.MEANSQUARE ) {
		m_MeanSquareMetric->SetFixedImageMask( m_MovingImageMaskObject );
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::PrintResults(
	void )
{
	std::cout << "Results of registration: " << std::endl << std::endl;

	if ( transform.VERSORRIGID ) {
		std::cout << "Versor x: " << m_RegistrationObject->GetLastTransformParameters()[0] << std::endl;
		std::cout << "Versor y: " << m_RegistrationObject->GetLastTransformParameters()[1] << std::endl;
		std::cout << "Versor z: " << m_RegistrationObject->GetLastTransformParameters()[2] << std::endl;
		std::cout << "Translation x: " << m_RegistrationObject->GetLastTransformParameters()[3] << std::endl;
		std::cout << "Translation y: " << m_RegistrationObject->GetLastTransformParameters()[4] << std::endl;
		std::cout << "Translation z: " << m_RegistrationObject->GetLastTransformParameters()[5] << std::endl;
	}

	if ( optimizer.REGULARSTEPGRADIENTDESCENT ) {
		std::cout << "Iterations: " << m_RegularStepGradientDescentOptimizer->GetCurrentIteration() << std::endl;
		std::cout << "Metric value: " << m_RegularStepGradientDescentOptimizer->GetValue() << std::endl;
	}

	if ( optimizer.VERSORRIGID3D ) {
		std::cout << "Iterations: " << m_VersorRigid3DTransformOptimizer->GetCurrentIteration() << std::endl;
		std::cout << "Metric value: " << m_VersorRigid3DTransformOptimizer->GetValue() << std::endl;
	}

	if ( optimizer.LBFGSBOPTIMIZER ) {
		std::cout << "Iterations: " << m_LBFGSBOptimizer->GetCurrentIteration() << std::endl;
		std::cout << "Metric value: " << m_RegistrationObject->GetMetric()->GetValue( m_RegistrationObject->GetLastTransformParameters() ) << std::endl;
	}

	if ( optimizer.AMOEBA ) {
		std::cout << "Metric value: " << m_AmoebaOptimizer->GetValue() << std::endl;
	}
}

template<class TFixedImageType, class TMovingImageType>
void RegistrationFactory3D<TFixedImageType, TMovingImageType>::StartRegistration(
	void )
{
	//set all parameters to make sure all user changes are noticed
	this->UpdateParameters();
	//check the image sizes and creat a joint image mask if the fixed image is bigger than the moving image
	//to avoid a itk sample error caused by a lack of spatial samples used by the metric

	m_observer = isis::extitk::IterationObserver::New();
	m_observer->setVerboseStep( UserOptions.SHOWITERATIONATSTEP );
	m_RegistrationObject->GetOptimizer()->AddObserver( itk::IterationEvent(), m_observer );
	m_RegistrationObject->SetNumberOfThreads( UserOptions.NumberOfThreads );

	try {
		m_RegistrationObject->StartRegistration();
	} catch ( itk::ExceptionObject &err ) {
		std::cerr << "isRegistrationFactory3D: Exception caught: " << std::endl << err << std::endl;
	}

	if ( UserOptions.PRINTRESULTS ) {
		this->PrintResults();
	}
}

} //end namespace Registration
} //end namespace isis
