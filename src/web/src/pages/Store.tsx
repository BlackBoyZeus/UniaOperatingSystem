import React, { useCallback, useEffect, useMemo, useState } from 'react';
import styled from '@emotion/styled';
import { useVirtualizer } from '@tanstack/virtual';
import { useInfiniteQuery } from '@tanstack/react-query';
import { usePerformanceMonitor } from '@performance-monitor/react';

import DashboardLayout from '../layouts/DashboardLayout';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import { GameService } from '../services/game.service';

// Interface for store items with HDR assets
interface StoreItem {
  id: string;
  title: string;
  description: string;
  price: number;
  imageUrl: string;
  hdrImageUrl: string;
  category: 'game' | 'addon' | 'content';
  releaseDate: Date;
  rating: number;
  fileSize: number;
  powerImpact: 'low' | 'medium' | 'high';
  fleetCompatible: boolean;
}

// GPU-accelerated styled components
const StoreContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 24px;
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
  
  /* GPU acceleration optimizations */
  transform: translateZ(0);
  backface-visibility: hidden;
  will-change: transform, opacity;
  
  /* Touch optimizations */
  touch-action: manipulation;
  overscroll-behavior: contain;
  
  /* HDR color space support */
  color-gamut: p3;
`;

const StoreHeader = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 0 24px;
  
  /* GPU acceleration */
  will-change: opacity;
  
  /* HDR support */
  color-gamut: p3;
  dynamic-range: high;
`;

const CategoryFilter = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  padding: 0 24px;
  
  /* Touch optimizations */
  touch-action: pan-x;
  overscroll-behavior: contain;
  will-change: transform;
`;

const Store: React.FC = React.memo(() => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [powerMode, setPowerMode] = useState<'performance' | 'balanced' | 'power-save'>('balanced');
  const { startMonitoring, metrics } = usePerformanceMonitor();
  const gameService = useMemo(() => new GameService(), []);

  // Initialize performance monitoring
  useEffect(() => {
    startMonitoring({
      maxLatency: 16, // Target 60fps
      sampleInterval: 1000,
      reportCallback: (metrics) => {
        if (metrics.averageLatency > 16) {
          setPowerMode('power-save');
        } else if (metrics.averageLatency < 8) {
          setPowerMode('performance');
        } else {
          setPowerMode('balanced');
        }
      }
    });
  }, [startMonitoring]);

  // Infinite query for store items with power-aware pagination
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetching
  } = useInfiniteQuery(
    ['store-items', selectedCategory],
    async ({ pageParam = 0 }) => {
      // Simulated API call
      return { items: [], nextPage: pageParam + 1 };
    },
    {
      getNextPageParam: (lastPage) => lastPage.nextPage,
      staleTime: powerMode === 'power-save' ? 5000 : 2000,
      cacheTime: powerMode === 'power-save' ? 300000 : 180000
    }
  );

  // Virtualized list for optimized rendering
  const parentRef = React.useRef<HTMLDivElement>(null);
  const allItems = useMemo(() => 
    data?.pages.flatMap(page => page.items) ?? [],
    [data]
  );

  const rowVirtualizer = useVirtualizer({
    count: allItems.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 300,
    overscan: powerMode === 'performance' ? 5 : 3
  });

  // Enhanced purchase handler with fleet compatibility check
  const handlePurchase = useCallback(async (item: StoreItem) => {
    try {
      const fleetStatus = await gameService.gameState$.value;
      
      if (item.fleetCompatible && fleetStatus?.currentFleet) {
        // Fleet-compatible purchase flow
        await gameService.emit('store:purchase', {
          itemId: item.id,
          fleetId: fleetStatus.currentFleet.id
        });
      } else {
        // Standard purchase flow
        await gameService.emit('store:purchase', {
          itemId: item.id
        });
      }
    } catch (error) {
      console.error('Purchase failed:', error);
    }
  }, [gameService]);

  return (
    <DashboardLayout>
      <StoreHeader>
        <h1>Store</h1>
        <Button
          variant="primary"
          powerSaveAware={powerMode === 'power-save'}
          hdrMode="auto"
        >
          Cart
        </Button>
      </StoreHeader>

      <CategoryFilter>
        {['all', 'games', 'addons', 'content'].map(category => (
          <Button
            key={category}
            variant={selectedCategory === category ? 'primary' : 'secondary'}
            powerSaveAware={powerMode === 'power-save'}
            hdrMode="auto"
            onClick={() => setSelectedCategory(category)}
          >
            {category.charAt(0).toUpperCase() + category.slice(1)}
          </Button>
        ))}
      </CategoryFilter>

      <div ref={parentRef} style={{ height: '100%', overflow: 'auto' }}>
        <StoreContainer
          style={{
            height: `${rowVirtualizer.getTotalSize()}px`,
            position: 'relative'
          }}
        >
          {rowVirtualizer.getVirtualItems().map((virtualItem) => {
            const item = allItems[virtualItem.index];
            return (
              <Card
                key={item.id}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualItem.size}px`,
                  transform: `translateY(${virtualItem.start}px)`
                }}
                variant="elevated"
                interactive
                hdrMode="auto"
                powerSaveMode={powerMode === 'power-save'}
                onClick={() => handlePurchase(item)}
              >
                <img
                  src={metrics?.hdrSupported ? item.hdrImageUrl : item.imageUrl}
                  alt={item.title}
                  loading="lazy"
                />
                <h3>{item.title}</h3>
                <p>{item.description}</p>
                <div>
                  <span>${item.price}</span>
                  {item.fleetCompatible && <span>Fleet Compatible</span>}
                </div>
              </Card>
            );
          })}
        </StoreContainer>
      </div>

      {hasNextPage && (
        <Button
          variant="secondary"
          powerSaveAware={powerMode === 'power-save'}
          hdrMode="auto"
          onClick={() => fetchNextPage()}
          disabled={isFetching}
        >
          Load More
        </Button>
      )}
    </DashboardLayout>
  );
});

Store.displayName = 'Store';

export default Store;