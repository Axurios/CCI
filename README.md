
# Setup
## Environment setup
Change in the 'env.ps1' file your Conda/Mamba path, then in command line :  <br>
windows : run  ` .\setup\windows\env.ps1 `  <br>
linux : run `bash .\setup\default\env.sh `  <br>
log in on Earth Engine following the instructions.  

## Dataset building
windows : run ` .\setup\windows\data.ps1 `  
linux : run `bash .\setup\linux\data.sh `

### Data stockage:
Google EarthEngine / GeoTessera  *(run once, slow, quota-limited)*  
  $\to$ .npy files on disk       *(cheap, permanent, version-controlled)*  
  $\to$ mmap'd Dataset           *(zero RAM overhead for tile storage)*  
  $\to$ patch sampler            *(random crops → huge effective dataset size)*  
  $\to$ DataLoader               *(transfer to the model/task during training)*

(once we run out data of our initial fecthing to train on, we can pull fresh data from new zones)


# Model training


# notes
year alignment (test-time usage)
directions :
alphaEarth combines more granular (monthly) division
geographical, random sampling in a defined zones, keep some zones (continent) for testing
clouds, biomes representativity

considers temporal info, biomass how temporal mosaic been considered (other dataset? land cover is snapshot)
biomass depends season, cci averaging
how data temporally in AlphaEarth
distribution oriented scatterplot

reconstructed and ground truth
don't want train input in the test visuals, few locations used as visualizations place (?)
diversified low biomass, high biomass

std average constraints


dataset map finishing
correlation between biomass and each of the biomass and embeddings (pca)
simples models 

major tom huggingface