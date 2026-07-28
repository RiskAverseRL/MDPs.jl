

@testset "Average - stationary matrix" begin
    
    P =
        [ 0.2 0.4 0.4 0.0 0.0 0.0;
          0.1 0.9 0.0 0.0 0.0 0.0;
          0.0 0.0 0.5 0.5 0.0 0.0;
          0.0 0.0 0.1 0.9 0.0 0.0;
          0.0 0.0 0.0 0.2 0.0 0.8; 
          0.0 0.0 0.0 0.0 0.0 1.0]

    Pstar = stationary_matrix(P)

    @test all(sum.(eachrow(Pstar)) .≈ 1.0)
    @test all(Pstar .≥ 0)
    @test all(Pstar * P .≈ Pstar)
    @test all(P * Pstar  .≈ Pstar)
end
