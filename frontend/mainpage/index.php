<!DOCTYPE html>
<?php 
$path = "../user_scores/";
if(isset($_POST["id_number"])&&isset($_POST["firstname"]))
{
  $dirname = $path.$_POST["firstname"]."_".$_POST["id_number"];

if (!is_file($dirname) && !is_dir($dirname)) {
    mkdir($dirname, 0777); 
	echo "<script type = 'text/javascript'>alert('You are registered');</script>";
}
else
{
	echo "<script type = 'text/javascript'>alert('You are already registered');</script>";
} 
}
else {
?>
	<a href="../registration/register.php">
<?php
}
?>
<html>
<head>
<title>Driving Test Results</title>
	<meta http-equiv="X-UA-Compatible" content="IE=EmulateIE7; IE=EmulateIE9">
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no"/>
    <link rel="shortcut icon" href=http://www.freshdesignweb.com/wp-content/themes/fv24/images/icon.ico />
    <link rel="stylesheet" type="text/css" href="styles.css" media="all" />
    <link rel="stylesheet" type="text/css" href="demo.css" media="all" />
	<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</head>
<script>
function myFunction(idf) {
  var popup = document.getElementById(idf);
  popup.classList.toggle("show");
}
</script>
<body>
<div class="container">
			<header>
				<h1>Your Driving Test Results<span>Name: <?php echo $_POST["firstname"] . " " . $_POST["lastname"]?></span><span>National Id Number:<?php echo $_POST["id_number"]?></h1>
            </header>   
		<div class="box">	
		<a class="button" href="#popupscore"> Click to see your Overall Score </a>
		</div>
	<header>
	<?php
		$foldername=$path.$_POST["firstname"]."_".$_POST["id_number"]; 
		$scoredv = -1;
		$scoredb = -1;
		$scoredt = -1;
		$immfail = false;
	?>
<div id="fdw-pricing-table">
	<div class="plan plan1">
        <div class="header">Lane Violations Count</div>
		<div class="monthly"> (Minor Fault) </div>
		<?php
		if(is_file($foldername."/laneviolations.txt"))
		{
		$myfile = fopen($foldername."/laneviolations.txt", "r");
		$float = (float)fgets($myfile);
		?>
		<div class="monthly"> unscored value :
		<?php echo $float; ?></div>  <!-- or print waiting until isn't available -->   
		<div class="monthly"> weightage -> 0.1 </div>
		<div class="monthly"> scored value : </div>
		<div class="price" name="laneviolations">
		<?php 
		$scoredv = max((10-$float*0.1),0);
		echo ($scoredv);  ?> </div>
		<?php
		}
		else { ?>
		<div class="monthly" name="laneviolations">waiting for scores...</div>
		<?php
		}
		?>
    </div>
    <div class="plan plan2">
        <div class="header">Behaviour Analysis Score</div>
		<div class="monthly"> (Minor Fault) </div>
		<?php
		if(is_file($foldername."/behaviouranalysis.txt"))
		{
		$myfile = fopen($foldername."/behaviouranalysis.txt", "r");
		$float = (float)fgets($myfile);
		?>
		<div class="monthly"> scored value : </div>
		<div class="price" name="behaviouranalysis">
		<?php 
		$scoredb = round(max($float,0),2); 
		echo $scoredb; ?> </div>
		<?php
		}
		else { ?>
		<div class="monthly" name="behaviouranalysis">waiting for scores...</div>
		<?php
		}
		?>
    </div>
    <div class="plan plan3">
        <div class="header">Traffic Cones Crashed Count</div>
		<div class="monthly"> (Major Fault) </div>
		<?php
		if(is_file($foldername."/conedetection.txt"))
		{
		$myfile = fopen($foldername."/conedetection.txt", "r");
		$float = (float)fgets($myfile);
		?>
        <div class="price" name="conehit">
		<?php echo $float; ?></div>  <!-- or print waiting until isn't available -->  
        <?php
		if($float==0) {
		?>
        <!--  tick or cross for depicting pass or fail -->
		<i class="fa fa-check-circle" style="font-size:30px;color:green"></i>
		<div class="monthly">PASS</div>
		<?php
		} else {
			$immfail=true;
		?>
		<i class="fa fa-times-circle" style="font-size:30px;color:red"></i>
		<div class="monthly">IMMEDIATE FAIL</div>
		<?php
		} }
		else { ?>
		<div class="monthly" name="conehit">waiting for scores...</div>
		<?php
		}
		?> 
    </div>
    <div class="plan plan4">
        <div class="header">Traffic Light Violation Check</div>
		<div class="monthly"> (Major Fault) </div>
		<?php
		if(is_file($foldername."/trafficlight.txt"))
		{
		$myfile = fopen($foldername."/trafficlight.txt", "r");
		$bool = fgets($myfile);
		?>
        <div class="price" name="trafficlight">
		<?php 
		if($bool=="true")
			echo "fail"; 
		else
			echo "pass";?></div>  <!-- or print waiting until isn't available --> 
        <?php
		if($bool=="false") {
		?>
        <!--  tick or cross for depicting pass or fail -->
		<i class="fa fa-check-circle" style="font-size:30px;color:green"></i>
		<div class="monthly">PASS</div>
		<?php
		} else {
			$immfail=true;
		?>
		<i class="fa fa-times-circle" style="font-size:30px;color:red"></i>
		<div class="monthly">Immediate FAIL</div>
		<?php
		} }
		else { ?>
		<div class="monthly" name="trafficlight">waiting for scores...</div>
		<?php
		}
		?>      
    </div>
	<div class="plan plan1">
        <div class="header">Trajectory Deviation Score</div>
		<div class="monthly"> (Minor Fault) </div>
		<?php
		if(is_file($foldername."/pathdeviation.txt"))
		{
		$myfile = fopen($foldername."/pathdeviation.txt", "r");
		$float = round((float)fgets($myfile),2);
		?>
		<div class="monthly"> unscored value :
		<?php echo $float; ?></div>  <!-- or print waiting until isn't available -->   
		<div class="monthly"> weightage -> 0.2 </div>
		<div class="monthly"> scored value : </div>
		<div class="price" name="pathdeviation">
		<?php 
		if($float>1000)
			$scoredt = max((10-($float- 1000)*0.2),0);
		else 
			$scoredt=10;
		echo ($scoredt);  ?> </div>
		<?php
		}
		else { ?>
		<div class="monthly" name="pathdeviation">waiting for scores...</div>
		<?php
		}
		?>
    </div>
</div>

<div id="popupscore" class="overlay">
<div class="popupscore">
		<h2>Your Overall Result</h2>
		<a class="close" href="#">&times;</a>
<?php	
fclose($myfile);
if($immfail==true){ ?>
		<i class="fa fa-times-circle" style="font-size:30px;color:red;"></i>
		<div class="content">
			Sorry!! You failed the test !! You did one or more major immediate fails!!
		</div>
</div>
<?php
}
else if($scoredv!=-1&&$scoredb!=-1&&$scoredt!=-1){
	$average=($scoredv+$scoredb+$scoredt)/3;
	if($average>=6){
	?>
	
	<i class="fa fa-check-circle" style="font-size:30px;color:green"></i>
		<div class="content">
			Congratulations!! You passed the test !!
		</div>
	</div>
	
	<?php
	}
	else {
	?>
	
	<i class="fa fa-times-circle" style="font-size:30px;color:red;"></i>
		<div class="content">
			Sorry!! You failed the test !! Threshold not satisfied!!
		</div>
	</div>
<?php
}}
else
{
?>
		<div class="content">
			Waiting for results!!!
		</div>
</div>
<?php
}
?>
</div>
</header>
</div>
</body>
</html>