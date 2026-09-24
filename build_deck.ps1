$ErrorActionPreference = 'Stop'

$base = (Get-Location).Path
$build = Join-Path $base '.codex-build'
$qa = Join-Path $build 'qa'
$assets = Join-Path $build 'assets'
$outDir = Join-Path $base 'output'
New-Item -ItemType Directory -Force -Path $build, $qa, $outDir | Out-Null
$outPptx = Join-Path $outDir 'Personalized_Career_Mentor_AI_Skill_Gap_Analysis_MSE1_v3.pptx'
$pdfPath = Join-Path $qa 'deck-review.pdf'
$bg = Join-Path $assets 'network-bg.png'
$hero = Join-Path $assets 'hero-ui.png'

function Rgb([int]$r,[int]$g,[int]$b) { return ($r + ($g -shl 8) + ($b -shl 16)) }
$C = @{
  Navy = (Rgb 15 23 52)
  Ink = (Rgb 26 34 60)
  Muted = (Rgb 93 108 142)
  Purple = (Rgb 91 67 242)
  Blue = (Rgb 38 142 218)
  Cyan = (Rgb 21 185 220)
  Mint = (Rgb 33 194 148)
  Pink = (Rgb 244 105 166)
  Red = (Rgb 236 73 104)
  Orange = (Rgb 238 143 34)
  Pale = (Rgb 247 249 255)
  White = (Rgb 255 255 255)
  Line = (Rgb 215 224 244)
  LightBlue = (Rgb 233 244 255)
  LightPurple = (Rgb 242 238 255)
  LightMint = (Rgb 230 250 246)
  LightPink = (Rgb 255 239 246)
  LightOrange = (Rgb 255 246 228)
}
$sw = 960
$sh = 540
$font = 'Aptos'
$fontDisplay = 'Aptos Display'

function AddBg($slide, [string]$mode = 'light') {
  if ($mode -eq 'light') {
    $pic = $slide.Shapes.AddPicture($bg, 0, -1, 0, 0, $sw, $sh)
    $pic.ZOrder(1)
    $bar = $slide.Shapes.AddShape(1, 0, 0, $sw, 4)
    $bar.Fill.ForeColor.RGB = $C.Purple
    $bar.Line.Visible = 0
  } else {
    $rect = $slide.Shapes.AddShape(1, 0, 0, $sw, $sh)
    $rect.Fill.ForeColor.RGB = $C.Navy
    $rect.Line.Visible = 0
  }
}

function AddText($slide, [string]$text, [double]$l, [double]$t, [double]$w, [double]$h, [double]$size = 18, $color = $C.Ink, [bool]$bold = $false, [string]$align = 'left', [string]$face = $font) {
  $text = $text.Replace('\n', [Environment]::NewLine)
  $shp = $slide.Shapes.AddTextbox(1, $l, $t, $w, $h)
  $tf = $shp.TextFrame2
  $tf.TextRange.Text = $text
  $tf.WordWrap = -1
  $tf.AutoSize = 0
  $tf.MarginLeft = 0
  $tf.MarginRight = 0
  $tf.MarginTop = 0
  $tf.MarginBottom = 0
  $tf.VerticalAnchor = 1
  $tf.TextRange.Font.Name = $face
  $tf.TextRange.Font.Size = $size
  $tf.TextRange.Font.Bold = $(if ($bold) { -1 } else { 0 })
  $tf.TextRange.Font.Fill.ForeColor.RGB = $color
  if ($align -eq 'center') { $tf.TextRange.ParagraphFormat.Alignment = 2 }
  elseif ($align -eq 'right') { $tf.TextRange.ParagraphFormat.Alignment = 3 }
  else { $tf.TextRange.ParagraphFormat.Alignment = 1 }
  return $shp
}

function AddBox($slide, [double]$l, [double]$t, [double]$w, [double]$h, $fill, $line = $C.Line, [double]$radius = 0, [double]$trans = 0) {
  $type = $(if ($radius -gt 0) { 5 } else { 1 })
  $shp = $slide.Shapes.AddShape($type, $l, $t, $w, $h)
  $shp.Fill.ForeColor.RGB = $fill
  $shp.Fill.Transparency = $trans
  if ($null -eq $line) { $shp.Line.Visible = 0 } else { $shp.Line.ForeColor.RGB = $line; $shp.Line.Weight = 1 }
  return $shp
}

function AddCircle($slide, [double]$l, [double]$t, [double]$d, $fill, $line = $null) {
  $shp = $slide.Shapes.AddShape(9, $l, $t, $d, $d)
  $shp.Fill.ForeColor.RGB = $fill
  if ($null -eq $line) { $shp.Line.Visible = 0 } else { $shp.Line.ForeColor.RGB = $line; $shp.Line.Weight = 1 }
  return $shp
}

function AddLine($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2, $color = $C.Line, [double]$weight = 1.5, [bool]$arrow = $false) {
  $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
  $line.Line.ForeColor.RGB = $color
  $line.Line.Weight = $weight
  if ($arrow) { $line.Line.EndArrowheadStyle = 2 }
  return $line
}

function AddPicture($slide, [string]$path, [double]$l, [double]$t, [double]$w, [double]$h) {
  $pic = $slide.Shapes.AddPicture($path, 0, -1, $l, $t, $w, $h)
  return $pic
}

function AddPill($slide, [string]$text, [double]$l, [double]$t, [double]$w, $fill = $C.LightPurple, $color = $C.Purple) {
  $shp = AddBox $slide $l $t $w 24 $fill $null 12
  AddText $slide $text ($l+10) ($t+4) ($w-20) 16 11 $color $true 'center'
}

function AddHeader($slide, [string]$kicker, [string]$title, [string]$subtitle = '') {
  AddText $slide $kicker.ToUpper() 48 26 260 16 11 $C.Purple $true
  AddText $slide $title 48 50 820 42 30 $C.Navy $true $fontDisplay
  if ($subtitle) { AddText $slide $subtitle 48 96 840 24 15 $C.Muted $false }
  AddLine $slide 48 128 912 128 $C.Line 1
}

function AddFooter($slide, [int]$num) {
  AddText $slide 'MSE-1 | KIET MCA | Personalized Career Mentor AI' 48 516 560 12 9 $C.Muted $false
  AddText $slide ("{0:00}" -f $num) 890 516 22 12 9 $C.Muted $true 'right'
}

function AddMetricTag($slide, [string]$metric, [double]$l = 820) {
  AddPill $slide $metric $l 26 92 $C.LightPurple $C.Purple
}

$pp = New-Object -ComObject PowerPoint.Application
$pp.Visible = -1
$pres = $pp.Presentations.Add()
$pres.PageSetup.SlideWidth = $sw
$pres.PageSetup.SlideHeight = $sh

function NewSlide([string]$bgMode = 'light') {
  $slide = $pres.Slides.Add($pres.Slides.Count + 1, 12)
  $slide.FollowMasterBackground = $false
  AddBg $slide $bgMode
  return $slide
}

# 1. Cover
$s = NewSlide 'light'
AddBox $s 48 28 26 26 $C.Purple $null 8
AddText $s '✦' 53 32 16 16 14 $C.White $true 'center'
AddText $s 'CAREER MENTOR AI' 84 33 180 18 12 $C.Navy $true
AddPill $s 'MSE-1 PROJECT-BASED LEARNING' 48 78 184 $C.LightPurple $C.Purple
AddText $s 'Personalized Career' 48 122 490 58 42 $C.Navy $true 'left' $fontDisplay
AddText $s 'Mentor AI' 48 174 360 58 44 $C.Blue $true 'left' $fontDisplay
AddText $s 'with Skill Gap Analysis' 48 226 500 42 29 $C.Purple $true 'left' $fontDisplay
AddText $s 'A role-specific career guide that compares resumes with job descriptions and turns gaps into clear next actions.' 50 288 450 60 17 $C.Muted $false
AddPill $s 'AI-powered web app + Chrome extension' 48 368 242 $C.LightMint $C.Mint
AddText $s 'Team' 48 424 60 16 11 $C.Purple $true
AddText $s 'Jatin Jangid  ·  Nitin  ·  Naina Vats  ·  Ilma Chaudhary' 48 446 450 20 14 $C.Navy $true
AddText $s '2228MCA0068   |   2628MCA0275   |   2628MCA0003   |   2628MCA0146' 48 470 450 16 10 $C.Muted $false
AddBox $s 568 78 344 382 $C.Navy $null 26
$heroPic = AddPicture $s $hero 576 86 328 366
AddText $s 'Resume Matcher AI' 592 98 220 16 12 $C.White $true
AddPill $s 'Powered by Groq' 780 98 104 $C.LightMint $C.Mint
AddText $s 'Compare. Understand. Improve.' 592 423 270 22 15 $C.White $true
AddText $s 'Front End Web Development (26CA205PCB)  |  MCA  |  KIET' 568 476 344 18 10 $C.Muted $false 'center'

# 2. Discovery
$s = NewSlide 'light'
AddHeader $s 'Metric 1 | Innovation & Problem Solving' 'User discovery & opportunity' 'The workflow is clear: job seekers need role-specific feedback before they apply.'
AddMetricTag $s 'METRIC 1'
AddText $s 'Primary users' 48 156 220 22 18 $C.Navy $true
AddText $s 'Students and freshers\nWorking professionals\nAnyone tailoring a resume to a specific role' 48 190 290 84 19 $C.Ink $false
AddText $s 'Observed friction' 48 304 220 22 18 $C.Navy $true
AddText $s 'Manual resume-to-JD comparison\nGeneric advice that ignores the target role\nUnclear gaps before an application\nCopy-paste effort across job boards' 48 338 325 110 17 $C.Ink $false
AddBox $s 444 156 420 292 $C.White $C.Line 24
AddText $s 'The opportunity' 476 184 260 26 24 $C.Purple $true $fontDisplay
AddText $s 'Make the “should I apply?” decision more informed.' 476 226 350 44 26 $C.Navy $true $fontDisplay
AddLine $s 476 286 824 286 $C.Line 1
AddCircle $s 486 316 34 $C.LightPurple
AddText $s '01' 494 325 18 16 11 $C.Purple $true 'center'
AddText $s 'Surface evidence from the resume' 536 317 280 22 17 $C.Ink $true
AddCircle $s 486 360 34 $C.LightBlue
AddText $s '02' 494 369 18 16 11 $C.Blue $true 'center'
AddText $s 'Compare against the target role' 536 361 280 22 17 $C.Ink $true
AddCircle $s 486 404 34 $C.LightMint
AddText $s '03' 494 413 18 16 11 $C.Mint $true 'center'
AddText $s 'Turn missing skills into actions' 536 405 280 22 17 $C.Ink $true
AddText $s 'Why it matters: job seekers receive a plan they can use immediately.' 48 474 760 20 14 $C.Muted $false
AddFooter $s 2

# 3. Problem statement
$s = NewSlide 'light'
AddHeader $s 'Metric 2 | Problem Statement' 'Problem definition' 'A precise problem statement anchors the product around one decision: fit for this role.'
AddMetricTag $s 'METRIC 2'
AddBox $s 48 154 864 108 $C.Navy $null 22
AddText $s 'Job seekers often apply without knowing how closely their resume matches a specific job, which skills are missing, or what to improve before applying.' 76 178 812 58 24 $C.White $true $fontDisplay
$items = @(
  @{k='WHO';v='Students, freshers, working professionals';c=$C.Purple},
  @{k='WHAT';v='Resume and job description do not line up clearly';c=$C.Blue},
  @{k='WHERE';v='Web applications and fast-moving job boards';c=$C.Cyan},
  @{k='WHEN';v='Before applying or preparing for an interview';c=$C.Mint},
  @{k='WHY';v='Manual comparison is slow and easy to miss';c=$C.Pink}
)
$x=48
foreach($it in $items){
  AddBox $s $x 296 162 136 $C.White $C.Line 16
  AddPill $s $it.k ($x+16) 314 62 $C.LightPurple $it.c
  AddText $s $it.v ($x+16) 350 130 58 15 $C.Ink $true
  $x += 176
}
AddText $s 'Product response' 48 458 140 18 12 $C.Purple $true
AddText $s 'Compare the inputs, explain the fit, and show what to do next.' 182 454 650 24 20 $C.Navy $true
AddFooter $s 3

# 4. Ideation
$s = NewSlide 'light'
AddHeader $s 'Metric 3 | Brainstorm & Prioritize' 'Ideation & prioritization' 'The chosen concept wins because it is relevant, feasible to build, and immediately actionable.'
AddMetricTag $s 'METRIC 3'
AddText $s 'Impact' 78 156 70 18 12 $C.Muted $true
AddText $s 'Low effort  →  High effort' 78 424 200 18 12 $C.Muted $true
AddLine $s 110 192 110 400 $C.Line 1.5
AddLine $s 110 400 416 400 $C.Line 1.5
AddText $s 'High' 72 204 40 18 12 $C.Muted $true
AddText $s 'Low' 72 382 40 18 12 $C.Muted $true
AddBox $s 154 236 174 72 $C.White $C.Line 16
AddText $s 'Generic resume\nchecker' 170 254 142 36 17 $C.Ink $true 'center'
AddBox $s 154 328 174 72 $C.LightPurple $C.Purple 16
AddText $s 'Resume + job\nmatcher' 170 346 142 36 18 $C.Purple $true 'center'
AddBox $s 354 186 174 72 $C.White $C.Line 16
AddText $s 'Course\nrecommender' 370 204 142 36 17 $C.Ink $true 'center'
AddBox $s 354 294 174 72 $C.White $C.Line 16
AddText $s 'LinkedIn\nassistant' 370 312 142 36 17 $C.Ink $true 'center'
AddText $s 'Prioritization lens' 582 158 240 22 21 $C.Navy $true
$criteria=@('Role relevance','Buildability with current stack','Actionable output','Privacy-aware workflow')
$y=204; $i=1
foreach($cr in $criteria){
  AddCircle $s 582 $y 28 $C.LightBlue
  AddText $s ("0$i") 588 ($y+7) 16 14 10 $C.Blue $true 'center'
  AddText $s $cr 626 ($y+2) 240 22 17 $C.Ink $true
  $y+=52; $i++
}
AddBox $s 582 414 300 50 $C.Navy $null 16
AddText $s 'Selected concept: Personalized Career Mentor AI' 600 430 266 20 15 $C.White $true 'center'
AddFooter $s 4

# 5. Journey
$s = NewSlide 'light'
AddHeader $s 'Metric 4 | Customer Journey' 'Customer journey' 'From a job posting to a focused improvement plan.'
AddMetricTag $s 'METRIC 4'
$steps=@(
  @{n='01';t='Find a role';d='Open a target job';c=$C.Purple},
  @{n='02';t='Provide inputs';d='Paste JD or upload PDF';c=$C.Blue},
  @{n='03';t='Run analysis';d='Choose depth and model';c=$C.Cyan},
  @{n='04';t='Read the fit';d='Score and skill groups';c=$C.Mint},
  @{n='05';t='Close gaps';d='Follow prioritized actions';c=$C.Orange},
  @{n='06';t='Prepare';d='Practice role questions';c=$C.Pink}
)
$x=70
for($i=0;$i -lt $steps.Count;$i++){
  $st=$steps[$i]
  if($i -lt $steps.Count-1){ AddLine $s ($x+84) 248 ($x+154) 248 $C.Line 2 $true }
  AddCircle $s $x 210 76 $st.c
  AddText $s $st.n ($x+18) 232 40 18 14 $C.White $true 'center'
  AddText $s $st.t ($x-8) 304 92 28 15 $C.Navy $true 'center'
  AddText $s $st.d ($x-18) 338 112 42 13 $C.Muted $false 'center'
  $x += 146
}
AddBox $s 84 410 792 60 $C.White $C.Line 18
AddText $s 'Pain point' 108 430 88 18 12 $C.Red $true
AddText $s '“I have a role in front of me, but I do not know what to fix first.”' 206 426 620 24 19 $C.Navy $true
AddText $s 'Product response: show the gap, rank the gap, and connect it to interview preparation.' 108 478 740 18 14 $C.Muted $false
AddFooter $s 5

# 6. Requirements
$s = NewSlide 'light'
AddHeader $s 'Metric 5 | Requirement Analysis' 'System requirements' 'The product requirements describe what the system does and how it should behave.'
AddMetricTag $s 'METRIC 5'
AddBox $s 48 154 412 320 $C.White $C.Line 22
AddText $s 'Functional requirements' 76 178 330 28 23 $C.Purple $true $fontDisplay
$fr=@('Accept pasted job description and resume text','Upload text-based PDF and extract readable text','Select model, analysis depth, and temperature','Return validated match, skill-gap, and interview fields','Download analysis as structured JSON')
$y=224; $i=1
foreach($v in $fr){ AddCircle $s 78 $y 24 $C.LightPurple; AddText $s ("0$i") 83 ($y+6) 14 12 9 $C.Purple $true 'center'; AddText $s $v 116 $y 314 28 14 $C.Ink $false; $y+=50; $i++ }
AddBox $s 500 154 412 320 $C.Navy $null 22
AddText $s 'Operational requirements' 528 178 330 28 23 $C.White $true $fontDisplay
$or=@('Responsive on desktop and mobile','Loading and error feedback during inference','API key stays in environment configuration','Modular components for maintainability','Manifest V3 compatibility for the extension')
$y=224; $i=1
foreach($v in $or){ AddCircle $s 530 $y 24 $C.Blue; AddText $s ("0$i") 535 ($y+6) 14 12 9 $C.White $true 'center'; AddText $s $v 568 $y 300 28 14 $C.White $false; $y+=50; $i++ }
AddFooter $s 6

# 7. System flow
$s = NewSlide 'light'
AddHeader $s 'Metric 5C | Technical Requirements' 'System flow & data movement' 'Both the website and Chrome extension feed the same analysis path.'
AddMetricTag $s 'METRIC 5C'
$nodes=@(
 @{x=48; y=220; w=134; h=78; title='Candidate'; sub='Resume + job description'; fill=$C.LightPurple; tc=$C.Purple},
 @{x=224; y=220; w=150; h=78; title='Next.js / React'; sub='Input + controls'; fill=$C.LightBlue; tc=$C.Blue},
 @{x=416; y=220; w=150; h=78; title='FastAPI'; sub='Analysis endpoint'; fill=$C.LightMint; tc=$C.Mint},
 @{x=608; y=220; w=150; h=78; title='LangChain + Groq'; sub='Role-constrained prompt'; fill=$C.LightOrange; tc=$C.Orange},
 @{x=800; y=220; w=112; h=78; title='Report'; sub='Validated JSON'; fill=$C.LightPink; tc=$C.Pink}
)
foreach($n in $nodes){ AddBox $s $n.x $n.y $n.w $n.h $n.fill $C.Line 16; AddText $s $n.title ($n.x+10) ($n.y+18) ($n.w-20) 20 16 $n.tc $true 'center'; AddText $s $n.sub ($n.x+10) ($n.y+46) ($n.w-20) 24 11 $C.Muted $false 'center' }
for($i=0;$i -lt $nodes.Count-1;$i++){ $a=$nodes[$i]; $b=$nodes[$i+1]; AddLine $s ($a.x+$a.w) 259 $b.x 259 $C.Purple 2 $true }
AddBox $s 224 346 150 62 $C.White $C.Line 16
AddText $s 'Optional PDF\ntext extraction' 238 361 122 34 15 $C.Navy $true 'center'
AddLine $s 299 346 299 298 $C.Blue 1.5 $true
AddBox $s 48 346 134 72 $C.White $C.Line 16
AddText $s 'Chrome extension\nLinkedIn job page' 60 360 110 46 12 $C.Navy $true 'center'
AddLine $s 115 346 115 298 $C.Pink 1.5 $true
AddText $s 'Pydantic schema validation keeps the AI response structured before it reaches the dashboard.' 48 446 820 24 16 $C.Muted $false
AddFooter $s 7

# 8. Model
$s = NewSlide 'light'
AddHeader $s 'Metric 6 | Interpretation & Modeling' 'System model & AI method' 'The design separates the user, application logic, and analysis data so the result stays explainable.'
AddMetricTag $s 'METRIC 6'
$cols=@(
 @{x=48; title='User model'; fill=$C.LightPurple; tc=$C.Purple; items=@('Resume text or PDF','Target job description','Model + depth controls')},
 @{x=330; title='Application model'; fill=$C.LightBlue; tc=$C.Blue; items=@('Responsive Next.js UI','FastAPI routes','Chrome extension entry')},
 @{x=612; title='Data model'; fill=$C.LightMint; tc=$C.Mint; items=@('Matched / missing skills','Score + assessments','Actions + interview prompts')}
)
foreach($col in $cols){
  AddBox $s $col.x 172 252 198 $col.fill $C.Line 22
  AddText $s $col.title ($col.x+22) 196 208 28 22 $col.tc $true $fontDisplay
  $y=246
  foreach($it in $col.items){ AddCircle $s ($col.x+24) $y 18 $C.White; AddText $s '✓' ($col.x+28) ($y+2) 10 12 9 $col.tc $true 'center'; AddText $s $it ($col.x+52) ($y-1) 180 22 15 $C.Ink $true; $y+=36 }
}
AddBox $s 48 404 816 56 $C.Navy $null 18
AddText $s 'Role-constrained prompt  →  Groq-hosted Llama model  →  JSON schema validation  →  action-oriented report' 72 423 768 20 16 $C.White $true 'center'
AddFooter $s 8

# 9. Product evidence
$s = NewSlide 'light'
AddHeader $s 'Interface & Output' 'Product evidence' 'The interface is designed to make a complex analysis easy to scan and act on.'
AddPill $s 'REPRESENTATIVE UI DIRECTION' 756 26 156 $C.LightBlue $C.Blue
AddBox $s 48 154 488 294 $C.Navy $null 22
$hero2=AddPicture $s $hero 56 162 472 278
AddText $s 'Resume Matcher AI' 76 176 210 16 12 $C.White $true
AddPill $s 'AI CAREER OPTIMIZATION' 76 404 146 $C.LightPurple $C.Purple
AddText $s 'A visual language inspired by the supplied project screens: light network background, rounded white cards, navy ink, and blue-violet accents.' 570 160 320 86 18 $C.Navy $true $fontDisplay
$outs=@(
 @{title='Match score'; sub='A single alignment signal for the role'; fill=$C.LightPurple; tc=$C.Purple},
 @{title='Skill groups'; sub='Matched, missing, and transferable skills'; fill=$C.LightMint; tc=$C.Mint},
 @{title='Action roadmap'; sub='Prioritized improvements and interview prep'; fill=$C.LightOrange; tc=$C.Orange}
)
$y=284
foreach($o in $outs){ AddBox $s 570 $y 320 58 $o.fill $C.Line 16; AddText $s $o.title 592 ($y+10) 126 18 16 $o.tc $true; AddText $s $o.sub 728 ($y+10) 142 34 12 $C.Muted $false; $y+=70 }
AddFooter $s 9

# 10. Stack
$s = NewSlide 'light'
AddHeader $s 'Implementation' 'Technology stack' 'A component-based client-server design connects the user experience to a validated AI service.'
AddPill $s 'BUILD & DEPLOY' 806 26 106 $C.LightMint $C.Mint
$stack=@(
 @{y=160; label='Frontend'; vals='Next.js 16  |  React 18  |  JSX/CSS  |  Framer Motion  |  Recharts'; c=$C.Purple; fill=$C.LightPurple},
 @{y=222; label='Backend'; vals='Python  |  FastAPI  |  Uvicorn  |  Pydantic  |  pypdf'; c=$C.Blue; fill=$C.LightBlue},
 @{y=284; label='AI service'; vals='LangChain  |  langchain-groq  |  Groq-hosted Llama models'; c=$C.Cyan; fill=$C.LightMint},
 @{y=346; label='Extension'; vals='Chrome Manifest V3  |  local storage  |  LinkedIn page extraction'; c=$C.Orange; fill=$C.LightOrange},
 @{y=408; label='Deployment'; vals='Vercel frontend config  |  Docker / Azure-oriented backend startup'; c=$C.Pink; fill=$C.LightPink}
)
foreach($st in $stack){ AddBox $s 48 $st.y 864 46 $st.fill $C.Line 14; AddPill $s $st.label 64 ($st.y+11) 104 $st.fill $st.c; AddText $s $st.vals 188 ($st.y+14) 696 18 15 $C.Navy $true }
AddText $s 'Repository structure keeps UI components, API routes, model schemas, extension files, and PDF parsing in focused modules.' 48 476 840 18 14 $C.Muted $false
AddFooter $s 10

# 11. Outcomes and team
$s = NewSlide 'light'
AddHeader $s 'Metric 7 | Complete Journey' 'From understanding to a working career mentor' 'The project connects a real user problem to a buildable, extensible solution.'
AddBox $s 48 156 864 74 $C.Navy $null 20
$journey=@('Understand','Define','Prioritize','Model','Build','Improve')
$x=82
for($i=0;$i -lt $journey.Count;$i++){
  AddText $s $journey[$i] $x 184 92 18 15 $C.White $true 'center'
  if($i -lt $journey.Count-1){ AddLine $s ($x+100) 192 ($x+126) 192 $C.Blue 1.5 $true }
  $x += 132
}
AddText $s 'Expected outcomes' 48 270 250 24 22 $C.Purple $true $fontDisplay
AddText $s '• Faster role fit checks before applying\n• Clear separation of strengths and gaps\n• Practical resume and interview recommendations\n• Chrome extension for faster LinkedIn analysis' 48 312 380 104 18 $C.Ink $false
AddText $s 'Future scope' 500 270 220 24 22 $C.Blue $true $fontDisplay
AddText $s '• Secure accounts and saved analyses\n• DOCX and scanned-PDF support\n• Explainable scoring with evidence\n• Learning roadmap and gap-closure analytics' 500 312 362 104 18 $C.Ink $false
AddBox $s 48 448 864 42 $C.LightPurple $null 16
AddText $s 'Team  |  Jatin Jangid 2228MCA0068  ·  Nitin 2628MCA0275  ·  Naina Vats 2628MCA0003  ·  Ilma Chaudhary 2628MCA0146' 64 461 832 16 12 $C.Navy $true 'center'
AddText $s 'Thank you' 48 498 200 20 14 $C.Purple $true
AddText $s 'Career guidance becomes more useful when it is specific, explainable, and actionable.' 246 496 620 22 15 $C.Muted $false 'right'
AddFooter $s 11

$pres.SaveAs($outPptx, 24)
try { $pres.ExportAsFixedFormat($pdfPath, 2, 0, 0, 1, 1, 1) } catch { }
for($i=1; $i -le $pres.Slides.Count; $i++){
  $png = Join-Path $qa ("slide-{0:00}.png" -f $i)
  $pres.Slides.Item($i).Export($png, 'PNG', 1600, 900)
}
$pres.Close()
$pp.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
Write-Output $outPptx
